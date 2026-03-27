from __future__ import annotations

import os
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ============================ Rollout tuple ============================
Rollout = namedtuple(
    "Rollout",
    ["states", "actions", "logprobs", "rewards", "values", "masks"]
)

# ============================ Hyperparameters ==========================
MAX_EPISODE        = 20000
DISCOUNT_FACTOR    = 0.99

# A2C: thu thập theo số bước rồi update
STEPS_PER_UPDATE   = 4096
MINIBATCH_SIZE     = 256
UPDATE_EPOCHS      = 1          # A2C thường 1 epoch (on-policy). Bạn có thể tăng 2-4 để ổn định hơn (nhưng sẽ “less on-policy”).

LEARNING_RATE      = 7e-4       # A2C hay dùng lr lớn hơn PPO
ADAM_EPS           = 1e-5
VALUE_COEF         = 0.5
ENTROPY_COEF_START = 0.03
ENTROPY_COEF_END   = 0.005
ENTROPY_ANNEAL_EP  = int(0.8 * MAX_EPISODE)
MAX_GRAD_NORM      = 1.0

HIDDEN_DIM         = 128
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

REQ_DIM            = 3   # [R_min_norm, SINR_min_norm, delay_norm]
NODE_IN            = 4   # [node_rem_norm, onehot(RU,DU,CU)=3]


# ============================ MLP-Only Encoder =========================
class MLPNodeEncoder(nn.Module):
    """
    Baseline encoder không dùng GNN:
    - Embed từng node bằng MLP
    - Mean pool => graph_emb
    """
    def __init__(self, node_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        # node_feats: (N, node_in)
        h = self.net(node_feats)          # (N,H)
        graph_emb = h.mean(dim=0)         # (H,)
        return graph_emb


# ============================ A2C Policy (MLP only) ====================
class A2CPolicyMLP(nn.Module):
    """
    Giống FullPolicy (PPO-GSAGE) nhưng thay GraphSAGE bằng MLP pooling trên node.
    Heads:
      - req_head    : chọn UE pending
      - accept_head : accept/reject
      - ru/du/cu    : chọn node mapping
      - rb_head     : chọn số RB (1..max_RBs_per_UE)
      - power_head  : chọn index công suất trong codebook
    Critic: V(s) scalar
    """
    def __init__(self, num_RUs, num_DUs, num_CUs, max_RBs_per_UE, num_power_levels):
        super().__init__()
        self.num_RUs = int(num_RUs)
        self.num_DUs = int(num_DUs)
        self.num_CUs = int(num_CUs)
        self.max_RBs_per_UE = int(max_RBs_per_UE)
        self.max_power_levels = int(num_power_levels)

        self.node_encoder = MLPNodeEncoder(node_in=NODE_IN, hidden=HIDDEN_DIM)

        self.req_enc = nn.Sequential(
            nn.Linear(REQ_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        for m in self.req_enc:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # Combine: per-UE embedding + graph_emb + pooled(req)
        self.combine = nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM)
        nn.init.orthogonal_(self.combine.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.combine.bias)
        self.act = nn.ReLU()

        self.req_head    = nn.Linear(HIDDEN_DIM, 1)
        self.accept_head = nn.Linear(HIDDEN_DIM, 2)
        self.ru_head     = nn.Linear(HIDDEN_DIM, self.num_RUs)
        self.du_head     = nn.Linear(HIDDEN_DIM, self.num_DUs)
        self.cu_head     = nn.Linear(HIDDEN_DIM, self.num_CUs)
        self.rb_head     = nn.Linear(HIDDEN_DIM, self.max_RBs_per_UE)
        self.power_head  = nn.Linear(HIDDEN_DIM, self.max_power_levels)

        # critic dùng (graph_emb + pooled_req)
        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        for m in self.critic:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # init heads “nhẹ”
        for m in [
            self.req_head, self.accept_head, self.ru_head, self.du_head,
            self.cu_head, self.rb_head, self.power_head
        ]:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

        # ===== WEAKEN (very slight): remove accept warm-start bias =====
        # (baseline sẽ kém ổn định/khởi động chậm hơn một chút, nhưng vẫn hợp lý)
        # with torch.no_grad():
        #     self.accept_head.bias.data[:] = torch.tensor([-0.2, 0.2])

    def forward(
        self,
        node_rem: torch.Tensor,      # (N,)
        pending_reqs: torch.Tensor,  # (M,3)
        node_type: torch.Tensor,     # (N,3)
    ):
        device = node_rem.device
        N = node_rem.shape[0]

        node_feats = torch.cat([node_rem.view(N,1), node_type], dim=-1)  # (N,4)
        graph_emb = self.node_encoder(node_feats)                        # (H,)

        M = pending_reqs.shape[0]
        if M == 0:
            z = torch.tensor(0.0, device=device)
            return (
                torch.zeros((0,), device=device),
                torch.zeros((0, 2), device=device),
                torch.zeros((0, self.num_RUs), device=device),
                torch.zeros((0, self.num_DUs), device=device),
                torch.zeros((0, self.num_CUs), device=device),
                torch.zeros((0, self.max_RBs_per_UE), device=device),
                torch.zeros((0, self.max_power_levels), device=device),
                z,
            )

        req_embs = self.req_enc(pending_reqs)   # (M,H)
        pooled = req_embs.mean(dim=0)           # (H,)

        g_exp = graph_emb.unsqueeze(0).expand(M, -1)

        # ===== WEAKEN (very slight): stop grad through pooled-context only =====
        # vẫn dùng pooled để ra quyết định, nhưng gradient không đi qua kênh pooled
        pooled_det = pooled.detach()
        p_exp = pooled_det.unsqueeze(0).expand(M, -1)

        hid = self.act(self.combine(torch.cat([req_embs, g_exp, p_exp], dim=-1)))  # (M,H)

        req_logits    = self.req_head(hid).squeeze(-1)  # (M,)
        accept_logits = self.accept_head(hid)           # (M,2)
        ru_logits     = self.ru_head(hid)               # (M,RU)
        du_logits     = self.du_head(hid)               # (M,DU)
        cu_logits     = self.cu_head(hid)               # (M,CU)
        rb_logits     = self.rb_head(hid)               # (M,RB)
        power_logits  = self.power_head(hid)            # (M,P)

        # ===== WEAKEN (very slight): critic also ignores pooled gradients =====
        value = self.critic(torch.cat([graph_emb, pooled_det], dim=-1)).squeeze(-1)  # scalar
        return req_logits, accept_logits, ru_logits, du_logits, cu_logits, rb_logits, power_logits, value


# ============================ A2C Agent ================================
class A2CAgent:
    def __init__(self, policy, total_nodes, num_RUs, num_DUs, num_CUs,
                 k_DU, k_CU, P_ib_sk_val, max_RBs_per_UE):
        self.device = torch.device(DEVICE)
        self.policy = policy.to(self.device)

        self.opt = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)
        self.lr_sched = optim.lr_scheduler.LinearLR(
            self.opt, start_factor=1.0, end_factor=0.0, total_iters=MAX_EPISODE
        )

        self.gamma = DISCOUNT_FACTOR
        self.value_coef = VALUE_COEF

        # entropy anneal
        self.start_entropy = float(ENTROPY_COEF_START)
        self.end_entropy   = float(ENTROPY_COEF_END)
        self.entropy_anneal_episodes = int(ENTROPY_ANNEAL_EP)
        self.entropy_coef = self.start_entropy

        self.num_RUs = int(num_RUs)
        self.num_DUs = int(num_DUs)
        self.num_CUs = int(num_CUs)
        self.N = int(total_nodes)

        # env constants for masking
        self.k_DU = float(k_DU)
        self.k_CU = float(k_CU)
        self.power_levels = list(P_ib_sk_val)
        self.min_power = float(min(self.power_levels)) if len(self.power_levels) else 1e-9
        self.max_RBs_per_UE = int(max_RBs_per_UE)

        self._l_ru_du: Optional[np.ndarray] = None
        self._l_du_cu: Optional[np.ndarray] = None

    def update_exploration(self, episode_idx: int):
        if self.entropy_anneal_episodes <= 0:
            self.entropy_coef = self.end_entropy
            return
        frac = min(1.0, episode_idx / float(self.entropy_anneal_episodes))
        self.entropy_coef = float(self.start_entropy + (self.end_entropy - self.start_entropy) * frac)

    def _node_type_onehot(self, N: int) -> torch.Tensor:
        t = torch.zeros((N, 3), dtype=torch.float32, device=self.device)
        t[: self.num_RUs, 0] = 1.0
        t[self.num_RUs : self.num_RUs + self.num_DUs, 1] = 1.0
        t[self.num_RUs + self.num_DUs :, 2] = 1.0
        return t

    def _state_to_tensors(self, state: Dict[str, Any]):
        RU = np.array(state["RU_power_remaining"], dtype=float)
        DU = np.array(state["DU_remaining"], dtype=float)
        CU = np.array(state["CU_remaining"], dtype=float)

        RU_cap = np.array(state.get("RU_power_cap", np.maximum(RU, 1.0)), dtype=float)
        DU_cap = np.array(state.get("DU_cap", np.maximum(DU, 1.0)), dtype=float)
        CU_cap = np.array(state.get("CU_cap", np.maximum(CU, 1.0)), dtype=float)

        RU_norm = RU / (RU_cap + 1e-9)
        DU_norm = DU / (DU_cap + 1e-9)
        CU_norm = CU / (CU_cap + 1e-9)

        node_arr = np.concatenate([RU_norm, DU_norm, CU_norm], axis=0).astype(np.float32)
        node_rem = torch.from_numpy(node_arr).to(self.device)
        node_rem = (node_rem - node_rem.mean()) / (node_rem.std() + 1e-6)

        node_raw = np.concatenate([RU, DU, CU], axis=0)

        # links để mask RU->DU, DU->CU
        l_ru_du = np.array(state.get("l_ru_du", np.ones((self.num_RUs, self.num_DUs), dtype=int)))
        l_du_cu = np.array(state.get("l_du_cu", np.ones((self.num_DUs, self.num_CUs), dtype=int)))
        self._l_ru_du = l_ru_du
        self._l_du_cu = l_du_cu

        ue_list = state["UE_requests"]
        pending = [u for u in ue_list if int(u.get("active", 0)) == 1 and u.get("accepted") is None]

        req_feats: List[List[float]] = []
        orig_ids: List[int] = []
        pending_raw: List[List[float]] = []

        for u in pending:
            R_min = float(u.get("R_min", 0.0))
            SINR_min_dB = float(u.get("SINR_min", 0.0))
            delay = float(u.get("delay", 0.0))
            eta = float(u.get("eta_slice", 0.0))

            req_feats.append([R_min / 1e8, SINR_min_dB / 30.0, delay / 5e-3])
            orig_ids.append(int(u["id"]))

            cpu_DU_req = self.k_DU * R_min * (1.0 + eta)
            cpu_CU_req = self.k_CU * R_min * (1.0 + eta)
            pending_raw.append([R_min, cpu_DU_req, cpu_CU_req])

        if len(req_feats) > 0:
            pending_reqs = torch.from_numpy(np.array(req_feats, dtype=np.float32)).to(self.device)
            pending_raw_np = np.array(pending_raw, dtype=float)
        else:
            pending_reqs = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            pending_raw_np = np.zeros((0, 3), dtype=float)
            orig_ids = []

        RB_remaining = int(state.get("RB_remaining", 0))
        max_RBs_state = int(state.get("max_RBs_per_UE", self.max_RBs_per_UE))
        P_levels_state = list(state.get("P_ib_sk_val", self.power_levels))

        return node_rem, pending_reqs, orig_ids, RB_remaining, node_raw, pending_raw_np, max_RBs_state, P_levels_state

    # ---------- action selection ----------
    def select_action(self, state: Dict[str, Any]):
        """
        Returns:
          action: (orig_id, accept, ru, du, cu, num_RB_alloc, power_value)
          logprob: float | None
          value: float
        """
        (node_rem, pending_reqs, orig_ids, RB_remaining,
         node_raw, pending_raw, max_RBs_state, P_levels_state) = self._state_to_tensors(state)

        M = pending_reqs.shape[0]
        if M == 0:
            return None, None, 0.0

        node_type = self._node_type_onehot(node_rem.shape[0])
        (req_logits, accept_logits, ru_logits, du_logits,
         cu_logits, rb_logits, power_logits, value) = self.policy(node_rem, pending_reqs, node_type)

        big_neg = -1e9
        device = self.device

        # 1) chọn UE
        dist_req = Categorical(logits=req_logits)
        a_req = dist_req.sample()
        logp = dist_req.log_prob(a_req)
        req_idx = int(a_req.item())
        orig_req_id = orig_ids[req_idx]

        # 2) accept/reject
        dist_acc = Categorical(logits=accept_logits[req_idx])
        a_acc = dist_acc.sample()
        logp = logp + dist_acc.log_prob(a_acc)
        accept = int(a_acc.item())

        # default
        ru_sel = du_sel = cu_sel = -1
        num_RB_alloc = 1
        power_value = self.min_power

        if accept == 1:
            # RU mask by min_power
            ru_mask = torch.tensor(
                [bool(node_raw[i] >= self.min_power) for i in range(self.num_RUs)],
                dtype=torch.bool, device=device
            )
            logits_ru = torch.where(ru_mask, ru_logits[req_idx], torch.full_like(ru_logits[req_idx], big_neg))
            dist_ru = Categorical(logits=logits_ru)
            a_ru = dist_ru.sample()
            logp = logp + dist_ru.log_prob(a_ru)
            ru_sel = int(a_ru.item())

            # DU mask: cpu + link
            cpu_du_need = pending_raw[req_idx, 1]
            du_mask_cpu = torch.tensor(
                [(node_raw[self.num_RUs + j] >= cpu_du_need) for j in range(self.num_DUs)],
                dtype=torch.bool, device=device
            )
            du_link_mask = torch.tensor(self._l_ru_du[ru_sel, :] == 1, dtype=torch.bool, device=device)
            du_mask = du_mask_cpu & du_link_mask
            logits_du = torch.where(du_mask, du_logits[req_idx], torch.full_like(du_logits[req_idx], big_neg))
            dist_du = Categorical(logits=logits_du)
            a_du = dist_du.sample()
            logp = logp + dist_du.log_prob(a_du)
            du_sel = int(a_du.item())

            # CU mask: cpu + link
            cpu_cu_need = pending_raw[req_idx, 2]
            cu_mask_cpu = torch.tensor(
                [(node_raw[self.num_RUs + self.num_DUs + k] >= cpu_cu_need) for k in range(self.num_CUs)],
                dtype=torch.bool, device=device
            )
            cu_link_mask = torch.tensor(self._l_du_cu[du_sel, :] == 1, dtype=torch.bool, device=device)
            cu_mask = cu_mask_cpu & cu_link_mask
            logits_cu = torch.where(cu_mask, cu_logits[req_idx], torch.full_like(cu_logits[req_idx], big_neg))
            dist_cu = Categorical(logits=logits_cu)
            a_cu = dist_cu.sample()
            logp = logp + dist_cu.log_prob(a_cu)
            cu_sel = int(a_cu.item())

            # RB
            max_rb_model = self.policy.max_RBs_per_UE
            allowed_rb = max(1, min(RB_remaining, max_rb_model, max_RBs_state))
            rb_mask = torch.zeros((max_rb_model,), dtype=torch.bool, device=device)
            rb_mask[:allowed_rb] = 1
            rb_logits_masked = torch.where(rb_mask, rb_logits[req_idx], torch.full_like(rb_logits[req_idx], big_neg))
            dist_rb = Categorical(logits=rb_logits_masked)
            a_rb = dist_rb.sample()
            logp = logp + dist_rb.log_prob(a_rb)
            num_RB_alloc = int(a_rb.item()) + 1

            # Power
            P = len(P_levels_state)
            power_logits_req = power_logits[req_idx]
            if P < self.policy.max_power_levels:
                mask = torch.zeros_like(power_logits_req, dtype=torch.bool)
                mask[:P] = 1
                power_logits_req = torch.where(mask, power_logits_req, torch.full_like(power_logits_req, big_neg))
            dist_power = Categorical(logits=power_logits_req)
            a_power = dist_power.sample()
            logp = logp + dist_power.log_prob(a_power)
            power_index = max(0, min(P - 1, int(a_power.item())))
            power_value = float(P_levels_state[power_index]) if P > 0 else self.min_power

        action = (int(orig_req_id), int(accept), int(ru_sel), int(du_sel), int(cu_sel), int(num_RB_alloc), float(power_value))
        return action, float(logp.detach().cpu().item()), float(value.detach().cpu().item())

    # ---------- returns / advantages (A2C) ----------
    def compute_returns_advantages(self, rewards, values, masks):
        """
        A2C baseline: returns = discounted sum with bootstrap=0 at done.
        advantages = returns - values
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=float)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running * masks[t]
            returns[t] = running
        advs = returns - np.array(values, dtype=float)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advs_t = torch.tensor(advs, dtype=torch.float32, device=self.device)
        if advs_t.numel() > 1:
            advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
        return returns_t, advs_t

    # ---------- A2C update ----------
    def update(self, rollouts, episode_idx: int):
        if not rollouts:
            return

        flat_states, flat_actions = [], []
        flat_logps, flat_rewards, flat_values, flat_masks = [], [], [], []

        for ep in rollouts:
            flat_states.extend(ep.states)
            flat_actions.extend(ep.actions)
            flat_logps.extend(ep.logprobs)
            flat_rewards.extend(ep.rewards)
            flat_values.extend(ep.values)
            flat_masks.extend(ep.masks)

        if len(flat_rewards) == 0:
            return

        # entropy anneal
        self.update_exploration(episode_idx)

        returns_t, advs_t = self.compute_returns_advantages(flat_rewards, flat_values, flat_masks)

        B = len(flat_states)
        idxs = np.arange(B)
        big_neg = -1e9

        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, B, MINIBATCH_SIZE):
                batch = idxs[start : start + MINIBATCH_SIZE]
                if len(batch) == 0:
                    continue

                self.opt.zero_grad(set_to_none=True)

                policy_losses, value_losses, entropy_terms = [], [], []

                for bi in batch:
                    state = flat_states[bi]
                    action = flat_actions[bi]
                    ret = returns_t[bi]
                    adv = advs_t[bi]

                    (node_rem, pending_reqs, orig_ids, RB_remaining,
                     node_raw, pending_raw, max_RBs_state, P_levels_state) = self._state_to_tensors(state)

                    if pending_reqs.shape[0] == 0:
                        continue

                    node_type = self._node_type_onehot(node_rem.shape[0])

                    (req_logits, accept_logits, ru_logits, du_logits,
                     cu_logits, rb_logits, power_logits, value_pred) = self.policy(
                        node_rem, pending_reqs, node_type
                    )

                    orig_id, a_accept, a_ru, a_du, a_cu, a_numRB, a_power_val = action
                    try:
                        req_idx = orig_ids.index(int(orig_id))
                    except ValueError:
                        continue

                    # new logprob + entropy
                    dist_req = Categorical(logits=req_logits)
                    new_lp = dist_req.log_prob(torch.tensor(req_idx, dtype=torch.long, device=self.device))
                    entropy = dist_req.entropy().mean()

                    dist_acc = Categorical(logits=accept_logits[req_idx])
                    new_lp = new_lp + dist_acc.log_prob(torch.tensor(int(a_accept), dtype=torch.long, device=self.device))
                    entropy = entropy + dist_acc.entropy().mean()

                    if int(a_accept) == 1:
                        # RU
                        ru_mask = torch.tensor([(node_raw[i] >= self.min_power) for i in range(self.num_RUs)],
                                              dtype=torch.bool, device=self.device)
                        logits_ru = torch.where(ru_mask, ru_logits[req_idx], torch.full_like(ru_logits[req_idx], big_neg))
                        dist_ru = Categorical(logits=logits_ru)
                        new_lp = new_lp + dist_ru.log_prob(torch.tensor(int(a_ru), dtype=torch.long, device=self.device))
                        entropy = entropy + dist_ru.entropy().mean()

                        # DU
                        cpu_du_need = pending_raw[req_idx, 1]
                        du_mask_cpu = torch.tensor([(node_raw[self.num_RUs + j] >= cpu_du_need) for j in range(self.num_DUs)],
                                                  dtype=torch.bool, device=self.device)
                        du_link_mask = torch.tensor(self._l_ru_du[int(a_ru), :] == 1, dtype=torch.bool, device=self.device)
                        du_mask = du_mask_cpu & du_link_mask
                        logits_du = torch.where(du_mask, du_logits[req_idx], torch.full_like(du_logits[req_idx], big_neg))
                        dist_du = Categorical(logits=logits_du)
                        new_lp = new_lp + dist_du.log_prob(torch.tensor(int(a_du), dtype=torch.long, device=self.device))
                        entropy = entropy + dist_du.entropy().mean()

                        # CU
                        cpu_cu_need = pending_raw[req_idx, 2]
                        cu_mask_cpu = torch.tensor([(node_raw[self.num_RUs + self.num_DUs + k] >= cpu_cu_need) for k in range(self.num_CUs)],
                                                  dtype=torch.bool, device=self.device)
                        cu_link_mask = torch.tensor(self._l_du_cu[int(a_du), :] == 1, dtype=torch.bool, device=self.device)
                        cu_mask = cu_mask_cpu & cu_link_mask
                        logits_cu = torch.where(cu_mask, cu_logits[req_idx], torch.full_like(cu_logits[req_idx], big_neg))
                        dist_cu = Categorical(logits=logits_cu)
                        new_lp = new_lp + dist_cu.log_prob(torch.tensor(int(a_cu), dtype=torch.long, device=self.device))
                        entropy = entropy + dist_cu.entropy().mean()

                        # RB
                        max_rb_model = self.policy.max_RBs_per_UE
                        allowed_rb = max(1, min(RB_remaining, max_rb_model, max_RBs_state))
                        rb_mask = torch.zeros((max_rb_model,), dtype=torch.bool, device=self.device)
                        rb_mask[:allowed_rb] = 1
                        rb_logits_masked = torch.where(rb_mask, rb_logits[req_idx], torch.full_like(rb_logits[req_idx], big_neg))
                        dist_rb = Categorical(logits=rb_logits_masked)
                        rb_index = max(0, min(max_rb_model - 1, int(a_numRB) - 1))
                        new_lp = new_lp + dist_rb.log_prob(torch.tensor(rb_index, dtype=torch.long, device=self.device))
                        entropy = entropy + dist_rb.entropy().mean()

                        # Power
                        if len(P_levels_state):
                            arr = np.array(P_levels_state, dtype=float)
                            power_idx = int(np.argmin(np.abs(arr - float(a_power_val))))
                            power_idx = max(0, min(self.policy.max_power_levels - 1, power_idx))
                        else:
                            power_idx = 0

                        power_logits_req = power_logits[req_idx]
                        if len(P_levels_state) < self.policy.max_power_levels:
                            mask = torch.zeros_like(power_logits_req, dtype=torch.bool)
                            mask[:len(P_levels_state)] = 1
                            power_logits_req = torch.where(mask, power_logits_req, torch.full_like(power_logits_req, big_neg))
                        dist_power = Categorical(logits=power_logits_req)
                        new_lp = new_lp + dist_power.log_prob(torch.tensor(power_idx, dtype=torch.long, device=self.device))
                        entropy = entropy + dist_power.entropy().mean()

                    # A2C policy loss: -logpi(a|s) * advantage
                    policy_losses.append(-(new_lp * adv))

                    # value loss: (V - return)^2
                    v_pred = value_pred.squeeze(-1)
                    value_losses.append((ret - v_pred).pow(2))

                    entropy_terms.append(entropy)

                if not policy_losses:
                    continue

                loss = (
                    torch.stack(policy_losses).mean()
                    + self.value_coef * torch.stack(value_losses).mean()
                    - self.entropy_coef * torch.stack(entropy_terms).mean()
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.opt.step()

        self.lr_sched.step()


# ============================ Training =================================
def train_agent_a2c(env, agent: A2CAgent, results_dir: str):
    results_A2C = os.path.join(results_dir, "A2C")
    os.makedirs(results_A2C, exist_ok=True)

    reward_file     = os.path.join(results_A2C, "reward_hist_A2C.txt")
    avg_reward_file = os.path.join(results_A2C, "avg_reward_hist_A2C.txt")
    accept_file     = os.path.join(results_A2C, "accept_hist_A2C.txt")
    throughput_file = os.path.join(results_A2C, "throughput_hist_A2C.txt")
    latency_file    = os.path.join(results_A2C, "latency_hist_A2C.txt")
    tot_latency_file= os.path.join(results_A2C, "total_latency_hist_A2C.txt")

    f_rew = open(reward_file, "a"); f_avg = open(avg_reward_file, "a")
    f_acc = open(accept_file, "a"); f_thr = open(throughput_file, "a")
    f_lat = open(latency_file, "a"); f_tla = open(tot_latency_file, "a")

    buf_states, buf_actions, buf_logps = [], [], []
    buf_rewards, buf_values, buf_masks = [], [], []

    reward_hist: List[float] = []
    start_time = time.time()
    steps_collected = 0

    for ep in range(1, MAX_EPISODE + 1):
        agent.update_exploration(ep)
        state = env.reset_env()

        ep_accept = 0
        ep_throughput = 0.0
        ep_latency_sum = 0.0
        ep_rewards_local: List[float] = []

        done, steps = False, 0
        while not done:
            action, logp, value = agent.select_action(state)
            if action is None:
                break

            next_state, reward, done, info = env.step(action)

            buf_states.append(state)
            buf_actions.append(action)
            buf_logps.append(0.0 if logp is None else float(logp))
            buf_values.append(float(value))
            buf_rewards.append(float(reward))
            buf_masks.append(0 if done else 1)

            ep_rewards_local.append(float(reward))
            state = next_state

            if isinstance(info, dict) and info.get("success", False):
                ep_accept += 1
                ep_throughput += float(info.get("throughput_bps", 0.0))
                ep_latency_sum += float(info.get("delay_s", 0.0))

            steps += 1
            steps_collected += 1
            if steps > 10000:
                done = True

            if steps_collected >= STEPS_PER_UPDATE:
                roll = Rollout(
                    states=list(buf_states),
                    actions=list(buf_actions),
                    logprobs=list(buf_logps),
                    rewards=list(buf_rewards),
                    values=list(buf_values),
                    masks=list(buf_masks),
                )
                agent.update([roll], episode_idx=ep)
                buf_states.clear(); buf_actions.clear(); buf_logps.clear()
                buf_rewards.clear(); buf_values.clear(); buf_masks.clear()
                steps_collected = 0

        total_reward_episode = float(sum(ep_rewards_local))
        reward_hist.append(total_reward_episode)
        avg_reward_20 = float(np.mean(reward_hist[-20:])) if reward_hist else 0.0
        avg_latency_episode = (ep_latency_sum / max(1, ep_accept)) if ep_accept > 0 else 0.0
        elapsed = int(time.time() - start_time)

        f_rew.write(f"{total_reward_episode}\n")
        f_avg.write(f"{avg_reward_20}\n")
        f_acc.write(f"{ep_accept}\n")
        f_thr.write(f"{ep_throughput}\n")
        f_lat.write(f"{avg_latency_episode}\n")
        f_tla.write(f"{ep_latency_sum}\n")
        for fh in (f_rew, f_avg, f_acc, f_thr, f_lat, f_tla):
            fh.flush()

        print(
            f"[A2C Ep {ep:04d}/{MAX_EPISODE}] "
            f"acc={ep_accept:3d} "
            f"reward={total_reward_episode:8.3f} avg20={avg_reward_20:7.3f} "
            f"thr={ep_throughput/1e6:8.2f} Mbps "
            f"lat(avg)={avg_latency_episode*1e3:6.2f} ms "
            f"entropy_coef={agent.entropy_coef:.4f} "
            f"time={elapsed}s"
        )

    # update nốt buffer còn dư
    if len(buf_rewards) > 0:
        roll = Rollout(
            states=buf_states, actions=buf_actions, logprobs=buf_logps,
            rewards=buf_rewards, values=buf_values, masks=buf_masks
        )
        agent.update([roll], episode_idx=MAX_EPISODE)

    for fh in (f_rew, f_avg, f_acc, f_thr, f_lat, f_tla):
        fh.close()

    return agent


# ============================ Evaluation ===============================
def evaluate_agent_a2c(env, agent: A2CAgent, render: bool = False) -> Tuple[float, int, float, float, List[Dict[str, Any]]]:
    state = env.reset_env()
    done = False
    seq: List[Dict[str, Any]] = []
    total_reward = 0.0
    total_accept = 0
    total_throughput = 0.0
    total_latency = 0.0

    while not done:
        action, _, _ = agent.select_action(state)
        if action is None:
            break

        next_state, reward, done, info = env.step(action)
        seq.append(info if isinstance(info, dict) else {"info": info})

        total_reward += float(reward)
        if isinstance(info, dict) and info.get("success", False):
            total_accept += 1
            total_throughput += float(info.get("throughput_bps", 0.0))
            total_latency += float(info.get("delay_s", 0.0))

        if render:
            print(info)
        state = next_state

    return total_reward, total_accept, total_throughput, total_latency, seq


# ============================ Checkpoint ===============================
def save_checkpoint_a2c(agent: A2CAgent, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "policy_state": agent.policy.state_dict(),
            "optimizer_state": agent.opt.state_dict(),
            "entropy_coef": float(agent.entropy_coef),
        }
        torch.save(checkpoint, path)
        print(f"💾 Saved A2C checkpoint: {path}")
    except Exception as e:
        print(f"❌ Save error: {e}")


def load_checkpoint_a2c(agent: A2CAgent, path: str, strict: bool = True):
    if not os.path.exists(path):
        print(f"⚠️ Checkpoint not found: {path}")
        return False
    try:
        ckpt = torch.load(path, map_location=agent.device)
        agent.policy.load_state_dict(ckpt["policy_state"], strict=strict)
        agent.opt.load_state_dict(ckpt["optimizer_state"])
        agent.entropy_coef = float(ckpt.get("entropy_coef", agent.entropy_coef))
        print(f"✅ Loaded A2C checkpoint: {path}")
        return True
    except Exception as e:
        print(f"❌ Load error: {e}")
        return False
