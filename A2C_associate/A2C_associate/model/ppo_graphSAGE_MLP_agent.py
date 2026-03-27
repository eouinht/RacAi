from __future__ import annotations

import os
import time
import copy
import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================ Hyperparameters ==========================
MAX_EPISODE        = 30000
DISCOUNT_FACTOR    = 0.99
GAE_LAMBDA         = 0.95

STEPS_PER_UPDATE   = 4096
MINIBATCH_SIZE     = 512          # tăng batch để tận dụng GPU tốt hơn (nếu VRAM ok)
LEARNING_RATE      = 3e-4
WEIGHT_DECAY       = 1e-4         # AdamW ổn định hơn
ADAM_EPS           = 1e-5

CLIP_RATIO         = 0.20
UPDATE_EPOCHS      = 4
VALUE_CLIP         = 0.20
VALUE_COEF         = 0.5

ENTROPY_COEF_START = 0.02
ENTROPY_COEF_END   = 0.003
ENTROPY_ANNEAL_EP  = int(0.8 * MAX_EPISODE)

TARGET_KL          = 0.01
MAX_GRAD_NORM      = 1.0

HIDDEN_DIM         = 128
ROUNDs             = 2
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
REQ_DIM            = 3

USE_AMP            = True if DEVICE == "cuda" else False
TORCH_BACKEND_BENCHMARK = True


# ============================ Utils ============================
def set_torch_speed_flags():
    torch.backends.cudnn.benchmark = TORCH_BACKEND_BENCHMARK
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def orth_init(m: nn.Module, gain: float):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)


# ============================ GraphSAGE ================================
class GraphSAGE(nn.Module):
    """
    GraphSAGE đơn giản: message/update lặp ROUNDs vòng.
    node_feats: (N, node_in)
    edge_feats: (N, N, edge_in)
    adj_mask:   (N, N) {0,1}
    """
    def __init__(self, node_in: int, edge_in: int):
        super().__init__()
        self.rounds = ROUNDs
        self.node_lin = nn.Linear(node_in, HIDDEN_DIM)
        self.edge_lin = nn.Linear(edge_in, HIDDEN_DIM)
        self.agg_lin = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.update_lin = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.act = nn.ReLU()

        orth_init(self.node_lin, math.sqrt(2))
        orth_init(self.edge_lin, math.sqrt(2))
        orth_init(self.agg_lin, math.sqrt(2))
        orth_init(self.update_lin, math.sqrt(2))

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor, adj_mask: torch.Tensor):
        h = self.act(self.node_lin(node_feats))   # (N,H)
        e = self.act(self.edge_lin(edge_feats))   # (N,N,H)
        N = h.shape[0]

        # mean aggregation theo degree
        deg = adj_mask.sum(dim=0).clamp(min=1.0).unsqueeze(-1)  # (N,1)

        for _ in range(self.rounds):
            sender = h.unsqueeze(1).expand(-1, N, -1)       # (N,N,H)
            msgs = torch.cat([sender, e], dim=-1)           # (N,N,2H)
            msgs = self.act(self.agg_lin(msgs))             # (N,N,H)
            msgs = msgs * adj_mask.unsqueeze(-1)            # mask edges
            agg = msgs.sum(dim=0) / deg                     # (N,H)
            h = self.act(self.update_lin(torch.cat([h, agg], dim=-1)))  # (N,H)

        graph_emb = h.mean(dim=0)                           # (H,)
        return h, graph_emb


# ============================ Policy ===================================
class FullPolicy(nn.Module):
    def __init__(self, num_RUs, num_DUs, num_CUs, max_RBs_per_UE, num_power_levels):
        super().__init__()
        self.num_RUs = int(num_RUs)
        self.num_DUs = int(num_DUs)
        self.num_CUs = int(num_CUs)
        self.max_RBs_per_UE = int(max_RBs_per_UE)
        self.max_power_levels = int(num_power_levels)

        node_in = 4
        edge_in = 1
        self.gnn = GraphSAGE(node_in=node_in, edge_in=edge_in)

        self.req_enc = nn.Sequential(
            nn.Linear(REQ_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        for m in self.req_enc:
            orth_init(m, math.sqrt(2))

        self.combine = nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM)
        orth_init(self.combine, math.sqrt(2))
        self.act = nn.ReLU()

        # heads
        self.req_head    = nn.Linear(HIDDEN_DIM, 1)
        self.accept_head = nn.Linear(HIDDEN_DIM, 2)
        self.ru_head     = nn.Linear(HIDDEN_DIM, self.num_RUs)
        self.du_head     = nn.Linear(HIDDEN_DIM, self.num_DUs)
        self.cu_head     = nn.Linear(HIDDEN_DIM, self.num_CUs)
        self.rb_head     = nn.Linear(HIDDEN_DIM, self.max_RBs_per_UE)
        self.power_head  = nn.Linear(HIDDEN_DIM, self.max_power_levels)

        for m in [self.req_head, self.accept_head, self.ru_head, self.du_head,
                  self.cu_head, self.rb_head, self.power_head]:
            orth_init(m, 0.01)

        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        for m in self.critic:
            orth_init(m, math.sqrt(2))

        # warm-start accept bias
        with torch.no_grad():
            self.accept_head.bias.data[:] = torch.tensor([-0.2, 0.2], device=self.accept_head.bias.device)

    def forward(self, node_rem, link, pending_reqs, node_type):
        device = node_rem.device
        N = node_rem.shape[0]

        node_feats = torch.cat([node_rem.view(N, 1), node_type], dim=-1)  # (N,4)
        edge_feats = link.view(N, N, 1)                                   # (N,N,1)
        adj_mask = (link > 0).float()

        _, graph_emb = self.gnn(node_feats, edge_feats, adj_mask)         # (H,)

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

        req_embs = self.req_enc(pending_reqs)  # (M,H)
        pooled = req_embs.mean(dim=0)          # (H,)
        g_exp = graph_emb.unsqueeze(0).expand(M, -1)
        p_exp = pooled.unsqueeze(0).expand(M, -1)

        hid = self.act(self.combine(torch.cat([req_embs, g_exp, p_exp], dim=-1)))

        req_logits    = self.req_head(hid).squeeze(-1)
        accept_logits = self.accept_head(hid)
        ru_logits     = self.ru_head(hid)
        du_logits     = self.du_head(hid)
        cu_logits     = self.cu_head(hid)
        rb_logits     = self.rb_head(hid)
        power_logits  = self.power_head(hid)

        value = self.critic(torch.cat([graph_emb, pooled], dim=-1)).squeeze(-1)
        return req_logits, accept_logits, ru_logits, du_logits, cu_logits, rb_logits, power_logits, value


# ============================ Rollout cached step ============================
@dataclass
class CachedState:
    node_rem: torch.Tensor          # (N,)
    link_t: torch.Tensor            # (N,N)
    pending_reqs: torch.Tensor      # (M,REQ_DIM)
    node_type: torch.Tensor         # (N,3)
    RB_remaining: int

    # raw info for masks
    node_raw: np.ndarray            # (N,) raw remaining
    pending_raw: np.ndarray         # (M,3) [Rmin, cpu_du, cpu_cu]
    l_ru_du: np.ndarray             # (RUs,DUs)
    l_du_cu: np.ndarray             # (DUs,CUs)
    power_levels: List[float]
    orig_ids: List[int]             # len M, mapping req_idx -> UE_id in env


@dataclass
class CachedAction:
    # env action
    env_action: Tuple[int, int, int, int, int, int, float]  # (UE_id, accept, ru, du, cu, numRB, power_value)

    # indices for PPO update (no searching)
    req_idx: int
    accept: int
    ru_idx: int
    du_idx: int
    cu_idx: int
    rb_idx: int            # 0..max_RBs_per_UE-1 (represents numRB = rb_idx+1)
    power_idx: int         # 0..max_power_levels-1


@dataclass
class RolloutBuffer:
    states: List[CachedState]
    actions: List[CachedAction]
    logprobs: List[float]
    rewards: List[float]
    values: List[float]
    next_values: List[float]
    masks: List[int]


# ============================ PPO Agent ================================
class PPOAgent:
    def __init__(self, policy: FullPolicy, total_nodes, num_RUs, num_DUs, num_CUs,
                 k_DU, k_CU, P_ib_sk_val, max_RBs_per_UE):
        set_torch_speed_flags()

        self.device = torch.device(DEVICE)
        self.policy = policy.to(self.device)

        self.opt = optim.AdamW(
            self.policy.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS, weight_decay=WEIGHT_DECAY
        )

        # scheduler: cosine decay theo episode (đủ ổn, nhẹ)
        self.lr_sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=MAX_EPISODE, eta_min=LEARNING_RATE * 0.05)

        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        self.gamma = DISCOUNT_FACTOR
        self.clip  = CLIP_RATIO
        self.epochs = UPDATE_EPOCHS
        self.value_coef = VALUE_COEF
        self.gae_lambda = GAE_LAMBDA

        self.num_RUs = int(num_RUs)
        self.num_DUs = int(num_DUs)
        self.num_CUs = int(num_CUs)
        self.N = int(total_nodes)

        self.start_entropy = float(ENTROPY_COEF_START)
        self.end_entropy   = float(ENTROPY_COEF_END)
        self.entropy_anneal_episodes = int(ENTROPY_ANNEAL_EP)
        self.entropy_coef = self.start_entropy

        self.k_DU = float(k_DU)
        self.k_CU = float(k_CU)
        self.power_levels_default = list(P_ib_sk_val)
        self.min_power = float(min(self.power_levels_default)) if len(self.power_levels_default) else 1e-9
        self.max_RBs_per_UE = int(max_RBs_per_UE)

        self.big_neg = -1e9

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

    def _state_to_cached(self, state: Dict[str, Any]) -> CachedState:
        # normalize remaining
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

        l_ru_du = np.array(state.get("l_ru_du", np.ones((self.num_RUs, self.num_DUs), dtype=int)))
        l_du_cu = np.array(state.get("l_du_cu", np.ones((self.num_DUs, self.num_CUs), dtype=int)))

        # build adjacency
        N = self.N
        link = np.zeros((N, N), dtype=np.float32)

        for i in range(self.num_RUs):
            gi = i
            for j in range(self.num_DUs):
                if l_ru_du[i, j] > 0:
                    gj = self.num_RUs + j
                    link[gi, gj] = 1.0
                    link[gj, gi] = 1.0

        for j in range(self.num_DUs):
            gi = self.num_RUs + j
            for k in range(self.num_CUs):
                if l_du_cu[j, k] > 0:
                    gj = self.num_RUs + self.num_DUs + k
                    link[gi, gj] = 1.0
                    link[gj, gi] = 1.0

        link_t = torch.from_numpy(link).to(self.device)

        # pending reqs
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

            # normalize: giữ giống bạn
            req_feats.append([R_min / 1e8, SINR_min_dB / 30.0, delay / 5e-3])
            orig_ids.append(int(u["id"]))

            cpu_DU_req = self.k_DU * R_min * (1.0 + eta)
            cpu_CU_req = self.k_CU * R_min * (1.0 + eta)
            pending_raw.append([R_min, cpu_DU_req, cpu_CU_req])

        if len(req_feats) > 0:
            pending_reqs = torch.from_numpy(np.array(req_feats, dtype=np.float32)).to(self.device)
            pending_raw_np = np.array(pending_raw, dtype=float)
        else:
            pending_reqs = torch.zeros((0, REQ_DIM), dtype=torch.float32, device=self.device)
            pending_raw_np = np.zeros((0, 3), dtype=float)
            orig_ids = []

        RB_remaining = int(state.get("RB_remaining", 0))
        power_levels = list(state.get("P_ib_sk_val", self.power_levels_default))

        node_type = self._node_type_onehot(node_rem.shape[0])

        return CachedState(
            node_rem=node_rem,
            link_t=link_t,
            pending_reqs=pending_reqs,
            node_type=node_type,
            RB_remaining=RB_remaining,
            node_raw=node_raw,
            pending_raw=pending_raw_np,
            l_ru_du=l_ru_du,
            l_du_cu=l_du_cu,
            power_levels=power_levels,
            orig_ids=orig_ids
        )

    @torch.no_grad()
    def get_value(self, cached: CachedState) -> float:
        if cached.pending_reqs.shape[0] == 0:
            return 0.0
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            *_, value = self.policy(cached.node_rem, cached.link_t, cached.pending_reqs, cached.node_type)
        return float(value.detach().float().cpu().item())

    @torch.no_grad()
    def select_action(self, state: Dict[str, Any]) -> Tuple[Optional[CachedState], Optional[CachedAction], Optional[float], float]:
        cached = self._state_to_cached(state)
        M = cached.pending_reqs.shape[0]
        if M == 0:
            return cached, None, None, 0.0

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            req_logits, accept_logits, ru_logits, du_logits, cu_logits, rb_logits, power_logits, value = \
                self.policy(cached.node_rem, cached.link_t, cached.pending_reqs, cached.node_type)

        # 1) select UE
        dist_req = Categorical(logits=req_logits)
        req_idx = int(dist_req.sample().item())
        logp = dist_req.log_prob(torch.tensor(req_idx, device=self.device)).float()
        entropy = dist_req.entropy().mean()

        ue_id = cached.orig_ids[req_idx]

        # 2) accept/reject
        dist_acc = Categorical(logits=accept_logits[req_idx])
        accept = int(dist_acc.sample().item())
        logp = logp + dist_acc.log_prob(torch.tensor(accept, device=self.device)).float()
        entropy = entropy + dist_acc.entropy().mean()

        # default (reject)
        ru_idx = du_idx = cu_idx = 0
        rb_idx = 0
        power_idx = 0
        env_ru = env_du = env_cu = -1
        env_numRB = 1
        env_power = float(self.min_power)

        if accept == 1:
            # RU mask by remaining power >= min_power
            ru_mask = torch.tensor(
                [bool(cached.node_raw[i] >= self.min_power) for i in range(self.num_RUs)],
                dtype=torch.bool, device=self.device
            )
            logits_ru = torch.where(ru_mask, ru_logits[req_idx], torch.full_like(ru_logits[req_idx], self.big_neg))
            dist_ru = Categorical(logits=logits_ru)
            ru_idx = int(dist_ru.sample().item())
            logp = logp + dist_ru.log_prob(torch.tensor(ru_idx, device=self.device)).float()
            entropy = entropy + dist_ru.entropy().mean()
            env_ru = ru_idx

            # DU mask by cpu + RU-DU link
            cpu_du_need = float(cached.pending_raw[req_idx, 1])
            du_mask_cpu = torch.tensor(
                [(cached.node_raw[self.num_RUs + j] >= cpu_du_need) for j in range(self.num_DUs)],
                dtype=torch.bool, device=self.device
            )
            du_link_mask = torch.tensor(cached.l_ru_du[ru_idx, :] == 1, dtype=torch.bool, device=self.device)
            du_mask = du_mask_cpu & du_link_mask
            logits_du = torch.where(du_mask, du_logits[req_idx], torch.full_like(du_logits[req_idx], self.big_neg))
            dist_du = Categorical(logits=logits_du)
            du_idx = int(dist_du.sample().item())
            logp = logp + dist_du.log_prob(torch.tensor(du_idx, device=self.device)).float()
            entropy = entropy + dist_du.entropy().mean()
            env_du = du_idx

            # CU mask by cpu + DU-CU link
            cpu_cu_need = float(cached.pending_raw[req_idx, 2])
            cu_mask_cpu = torch.tensor(
                [(cached.node_raw[self.num_RUs + self.num_DUs + k] >= cpu_cu_need) for k in range(self.num_CUs)],
                dtype=torch.bool, device=self.device
            )
            cu_link_mask = torch.tensor(cached.l_du_cu[du_idx, :] == 1, dtype=torch.bool, device=self.device)
            cu_mask = cu_mask_cpu & cu_link_mask
            logits_cu = torch.where(cu_mask, cu_logits[req_idx], torch.full_like(cu_logits[req_idx], self.big_neg))
            dist_cu = Categorical(logits=logits_cu)
            cu_idx = int(dist_cu.sample().item())
            logp = logp + dist_cu.log_prob(torch.tensor(cu_idx, device=self.device)).float()
            entropy = entropy + dist_cu.entropy().mean()
            env_cu = cu_idx

            # RB mask by remaining
            max_rb_model = self.policy.max_RBs_per_UE
            allowed_rb = max(1, min(cached.RB_remaining, max_rb_model))
            rb_mask = torch.zeros((max_rb_model,), dtype=torch.bool, device=self.device)
            rb_mask[:allowed_rb] = 1
            logits_rb = torch.where(rb_mask, rb_logits[req_idx], torch.full_like(rb_logits[req_idx], self.big_neg))
            dist_rb = Categorical(logits=logits_rb)
            rb_idx = int(dist_rb.sample().item())  # 0..max_rb-1
            logp = logp + dist_rb.log_prob(torch.tensor(rb_idx, device=self.device)).float()
            entropy = entropy + dist_rb.entropy().mean()
            env_numRB = rb_idx + 1

            # power mask by available power levels length
            P = len(cached.power_levels)
            logits_pw = power_logits[req_idx]
            if P < self.policy.max_power_levels:
                mask = torch.zeros_like(logits_pw, dtype=torch.bool)
                mask[:P] = 1
                logits_pw = torch.where(mask, logits_pw, torch.full_like(logits_pw, self.big_neg))
            dist_pw = Categorical(logits=logits_pw)
            power_idx = int(dist_pw.sample().item())
            power_idx = max(0, min(P - 1, power_idx)) if P > 0 else 0
            logp = logp + dist_pw.log_prob(torch.tensor(power_idx, device=self.device)).float()
            entropy = entropy + dist_pw.entropy().mean()
            env_power = float(cached.power_levels[power_idx]) if P > 0 else float(self.min_power)

        env_action = (int(ue_id), int(accept), int(env_ru), int(env_du), int(env_cu), int(env_numRB), float(env_power))
        cached_action = CachedAction(
            env_action=env_action,
            req_idx=req_idx,
            accept=accept,
            ru_idx=ru_idx,
            du_idx=du_idx,
            cu_idx=cu_idx,
            rb_idx=rb_idx,
            power_idx=power_idx
        )
        return cached, cached_action, float(logp.detach().cpu().item()), float(value.detach().float().cpu().item())

    # ---------------- GAE/returns ----------------
    def compute_returns_advantages(self, rewards, values, next_values, masks):
        gamma, lam = self.gamma, self.gae_lambda
        T = len(rewards)
        advs = np.zeros(T, dtype=np.float32)
        lastgae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_values[t] * masks[t] - values[t]
            lastgae = delta + gamma * lam * masks[t] * lastgae
            advs[t] = lastgae
        returns = advs + np.array(values, dtype=np.float32)
        return returns, advs

    # ---------------- PPO update (fast: reuse cached tensors) ----------------
    def update(self, buf: RolloutBuffer, episode_idx: int):
        B = len(buf.rewards)
        if B == 0:
            return

        self.update_exploration(episode_idx)

        rewards = np.array(buf.rewards, dtype=np.float32)
        values  = np.array(buf.values, dtype=np.float32)
        next_v  = np.array(buf.next_values, dtype=np.float32)
        masks   = np.array(buf.masks, dtype=np.float32)

        returns, advs = self.compute_returns_advantages(rewards, values, next_v, masks)

        # normalize advantage global (ổn định), và cũng normalize theo minibatch sau nữa
        advs_t = torch.from_numpy(advs).to(self.device)
        if advs_t.numel() > 1:
            advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
        returns_t = torch.from_numpy(returns).to(self.device)

        old_logps_t = torch.tensor(buf.logprobs, dtype=torch.float32, device=self.device)
        values_old_t = torch.tensor(buf.values, dtype=torch.float32, device=self.device)

        idxs = np.arange(B)

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            kl_list = []

            for start in range(0, B, MINIBATCH_SIZE):
                batch = idxs[start:start + MINIBATCH_SIZE]
                if len(batch) == 0:
                    continue

                # minibatch normalize advantage (thường giúp ổn định hơn trong PPO)
                adv_mb = advs_t[batch]
                if adv_mb.numel() > 1:
                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                self.opt.zero_grad(set_to_none=True)

                policy_losses = []
                value_losses  = []
                entropies     = []
                kls           = []

                # loop per-sample (do variable M per state); nhưng KHÔNG encode state lại -> nhanh hơn nhiều
                for ii, bi in enumerate(batch):
                    s = buf.states[bi]
                    a = buf.actions[bi]
                    old_lp = old_logps_t[bi]
                    ret = returns_t[bi]
                    adv = adv_mb[ii]
                    v_old = values_old_t[bi]

                    if s.pending_reqs.shape[0] == 0:
                        continue

                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        req_logits, accept_logits, ru_logits, du_logits, cu_logits, rb_logits, power_logits, v_pred = \
                            self.policy(s.node_rem, s.link_t, s.pending_reqs, s.node_type)

                        # build new logprob from stored indices (NO searching)
                        req_idx = a.req_idx
                        dist_req = Categorical(logits=req_logits)
                        new_lp = dist_req.log_prob(torch.tensor(req_idx, device=self.device)).float()
                        entropy = dist_req.entropy().mean()

                        dist_acc = Categorical(logits=accept_logits[req_idx])
                        new_lp = new_lp + dist_acc.log_prob(torch.tensor(a.accept, device=self.device)).float()
                        entropy = entropy + dist_acc.entropy().mean()

                        if a.accept == 1:
                            # RU masked
                            ru_mask = torch.tensor(
                                [bool(s.node_raw[i] >= self.min_power) for i in range(self.num_RUs)],
                                dtype=torch.bool, device=self.device
                            )
                            logits_ru = torch.where(ru_mask, ru_logits[req_idx], torch.full_like(ru_logits[req_idx], self.big_neg))
                            dist_ru = Categorical(logits=logits_ru)
                            new_lp = new_lp + dist_ru.log_prob(torch.tensor(a.ru_idx, device=self.device)).float()
                            entropy = entropy + dist_ru.entropy().mean()

                            # DU masked
                            cpu_du_need = float(s.pending_raw[req_idx, 1])
                            du_mask_cpu = torch.tensor(
                                [(s.node_raw[self.num_RUs + j] >= cpu_du_need) for j in range(self.num_DUs)],
                                dtype=torch.bool, device=self.device
                            )
                            du_link_mask = torch.tensor(s.l_ru_du[a.ru_idx, :] == 1, dtype=torch.bool, device=self.device)
                            du_mask = du_mask_cpu & du_link_mask
                            logits_du = torch.where(du_mask, du_logits[req_idx], torch.full_like(du_logits[req_idx], self.big_neg))
                            dist_du = Categorical(logits=logits_du)
                            new_lp = new_lp + dist_du.log_prob(torch.tensor(a.du_idx, device=self.device)).float()
                            entropy = entropy + dist_du.entropy().mean()

                            # CU masked
                            cpu_cu_need = float(s.pending_raw[req_idx, 2])
                            cu_mask_cpu = torch.tensor(
                                [(s.node_raw[self.num_RUs + self.num_DUs + k] >= cpu_cu_need) for k in range(self.num_CUs)],
                                dtype=torch.bool, device=self.device
                            )
                            cu_link_mask = torch.tensor(s.l_du_cu[a.du_idx, :] == 1, dtype=torch.bool, device=self.device)
                            cu_mask = cu_mask_cpu & cu_link_mask
                            logits_cu = torch.where(cu_mask, cu_logits[req_idx], torch.full_like(cu_logits[req_idx], self.big_neg))
                            dist_cu = Categorical(logits=logits_cu)
                            new_lp = new_lp + dist_cu.log_prob(torch.tensor(a.cu_idx, device=self.device)).float()
                            entropy = entropy + dist_cu.entropy().mean()

                            # RB masked
                            max_rb_model = self.policy.max_RBs_per_UE
                            allowed_rb = max(1, min(s.RB_remaining, max_rb_model))
                            rb_mask = torch.zeros((max_rb_model,), dtype=torch.bool, device=self.device)
                            rb_mask[:allowed_rb] = 1
                            logits_rb = torch.where(rb_mask, rb_logits[req_idx], torch.full_like(rb_logits[req_idx], self.big_neg))
                            dist_rb = Categorical(logits=logits_rb)
                            new_lp = new_lp + dist_rb.log_prob(torch.tensor(a.rb_idx, device=self.device)).float()
                            entropy = entropy + dist_rb.entropy().mean()

                            # Power masked
                            P = len(s.power_levels)
                            logits_pw = power_logits[req_idx]
                            if P < self.policy.max_power_levels:
                                mask = torch.zeros_like(logits_pw, dtype=torch.bool)
                                mask[:P] = 1
                                logits_pw = torch.where(mask, logits_pw, torch.full_like(logits_pw, self.big_neg))
                            dist_pw = Categorical(logits=logits_pw)
                            new_lp = new_lp + dist_pw.log_prob(torch.tensor(a.power_idx, device=self.device)).float()
                            entropy = entropy + dist_pw.entropy().mean()

                        # PPO ratio
                        ratio = torch.exp(new_lp - old_lp)
                        surr1 = ratio * adv
                        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                        p_loss = -torch.min(surr1, surr2)

                        # value loss (clipped)
                        v_pred = v_pred.squeeze(-1)
                        v_clip = v_old + torch.clamp(v_pred - v_old, -VALUE_CLIP, VALUE_CLIP)
                        vloss1 = (ret - v_pred).pow(2)
                        vloss2 = (ret - v_clip).pow(2)
                        v_loss = torch.max(vloss1, vloss2)

                        # approx KL
                        approx_kl = (old_lp - new_lp).detach()

                    policy_losses.append(p_loss)
                    value_losses.append(v_loss)
                    entropies.append(entropy)
                    kls.append(approx_kl)

                if not policy_losses:
                    continue

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    loss = (
                        torch.stack(policy_losses).mean()
                        + self.value_coef * torch.stack(value_losses).mean()
                        - self.entropy_coef * torch.stack(entropies).mean()
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.scaler.step(self.opt)
                self.scaler.update()

                if kls:
                    kl_list.append(torch.stack(kls).mean().item())

            if kl_list:
                mean_kl = float(np.mean(kl_list))
                if mean_kl > 1.5 * TARGET_KL:
                    break

        self.lr_sched.step()


# ============================ Training =================================
def train_agent(env, agent: PPOAgent, results_dir: str):
    results_PPO = os.path.join(results_dir, "PPO")
    os.makedirs(results_PPO, exist_ok=True)

    reward_file      = os.path.join(results_PPO, "reward_hist_PPO.txt")
    avg_reward_file  = os.path.join(results_PPO, "avg_reward_hist_PPO.txt")
    accept_file      = os.path.join(results_PPO, "accept_hist_PPO.txt")
    throughput_file  = os.path.join(results_PPO, "throughput_hist_PPO.txt")
    latency_file     = os.path.join(results_PPO, "latency_hist_PPO.txt")
    tot_latency_file = os.path.join(results_PPO, "total_latency_hist_PPO.txt")

    f_rew = open(reward_file, "a"); f_avg = open(avg_reward_file, "a")
    f_acc = open(accept_file, "a"); f_thr = open(throughput_file, "a")
    f_lat = open(latency_file, "a"); f_tla = open(tot_latency_file, "a")

    reward_hist: List[float] = []
    start_time = time.time()

    # rollout buffer
    buf = RolloutBuffer(states=[], actions=[], logprobs=[], rewards=[], values=[], next_values=[], masks=[])

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
            cached_s, cached_a, logp, value = agent.select_action(state)
            if cached_a is None:
                break

            next_state, reward, done, info = env.step(cached_a.env_action)

            # bootstrap next value using cached next state (fast)
            cached_ns = agent._state_to_cached(next_state)
            v_next = agent.get_value(cached_ns) if not done else 0.0

            buf.states.append(cached_s)
            buf.actions.append(cached_a)
            buf.logprobs.append(0.0 if logp is None else float(logp))
            buf.values.append(float(value))
            buf.next_values.append(float(v_next))
            buf.rewards.append(float(reward))
            buf.masks.append(0 if done else 1)

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
                agent.update(buf, episode_idx=ep)
                # clear buffer
                buf = RolloutBuffer(states=[], actions=[], logprobs=[], rewards=[], values=[], next_values=[], masks=[])
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
            f"[PPO-GSAGE Ep {ep:04d}/{MAX_EPISODE}] "
            f"acc={ep_accept:3d} "
            f"reward={total_reward_episode:8.3f} avg20={avg_reward_20:7.3f} "
            f"thr={ep_throughput/1e6:8.2f} Mbps "
            f"lat(avg)={avg_latency_episode*1e3:6.2f} ms "
            f"entropy_coef={agent.entropy_coef:.4f} "
            f"time={elapsed}s"
        )

    # final update if leftover
    if len(buf.rewards) > 0:
        agent.update(buf, episode_idx=MAX_EPISODE)

    for fh in (f_rew, f_avg, f_acc, f_thr, f_lat, f_tla):
        fh.close()

    return agent


# ============================ Evaluation ===============================
@torch.no_grad()
def evaluate_agent(env, agent: PPOAgent, render: bool = False) -> Tuple[float, int, float, float, List[Dict[str, Any]]]:
    state = env.reset_env()
    done = False
    seq: List[Dict[str, Any]] = []
    total_reward = 0.0
    total_accept = 0
    total_throughput = 0.0
    total_latency = 0.0

    while not done:
        cached_s, cached_a, _, _ = agent.select_action(state)
        if cached_a is None:
            break

        next_state, reward, done, info = env.step(cached_a.env_action)
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
def save_checkpoint(agent: PPOAgent, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "policy_state": agent.policy.state_dict(),
            "optimizer_state": agent.opt.state_dict(),
            "scaler_state": agent.scaler.state_dict() if agent.scaler is not None else None,
            "entropy_coef": float(agent.entropy_coef),
        }
        torch.save(checkpoint, path)
        print(f"💾 Đã lưu checkpoint tại: {path}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu checkpoint: {e}")


def load_checkpoint(agent: PPOAgent, path: str, strict: bool = True):
    if not os.path.exists(path):
        print(f"⚠️ Không tìm thấy checkpoint tại: {path}")
        return False
    try:
        ckpt = torch.load(path, map_location=agent.device)
        agent.policy.load_state_dict(ckpt["policy_state"], strict=strict)
        agent.opt.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state", None) is not None and agent.scaler is not None:
            agent.scaler.load_state_dict(ckpt["scaler_state"])
        agent.entropy_coef = float(ckpt.get("entropy_coef", agent.entropy_coef))
        print(f"✅ Đã load checkpoint từ: {path}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi load checkpoint từ {path}: {e}")
        return False
