from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ppo.encoder import EncoderConfig, ObservationEncoder


@dataclass
class ModelConfig:
    ue_feat_dim: int
    cell_feat_dim: int
    hidden_dim: int = 128
    latent_dim: int = 128


class Actor_Critic(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = ObservationEncoder(
            EncoderConfig(
                ue_feat_dim=cfg.ue_feat_dim,
                cell_feat_dim=cfg.cell_feat_dim,
                hidden_dim=cfg.hidden_dim,
                latent_dim=cfg.latent_dim,
            )
        )

        # PHẢI là latent_dim * 3 vì ghép:
        # ue_latent + cell_latent + global_latent
        self.actor_pair = nn.Sequential(
            nn.Linear(cfg.latent_dim * 3, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

        self.critic = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def action_logits(
        self,
        ue_latent: torch.Tensor,
        cell_latent: torch.Tensor,
        global_latent: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_ues, dim = ue_latent.shape
        num_cells = cell_latent.shape[1]

        ue_expand = ue_latent.unsqueeze(2).expand(bsz, num_ues, num_cells, dim)
        cell_expand = cell_latent.unsqueeze(1).expand(bsz, num_ues, num_cells, dim)
        global_expand = global_latent.unsqueeze(1).unsqueeze(2).expand(
            bsz, num_ues, num_cells, dim
        )

        pair_feat = torch.cat([ue_expand, cell_expand, global_expand], dim=-1)
        logits = self.actor_pair(pair_feat).squeeze(-1)

        invalid = cell_mask.unsqueeze(1) <= 0
        logits = logits.masked_fill(invalid, -1e9)
        return logits

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(obs)

        logits = self.action_logits(
            ue_latent=enc["ue_latent"],
            cell_latent=enc["cell_latent"],
            global_latent=enc["global_latent"],
            cell_mask=obs["cell_mask"],
        )
        values = self.critic(enc["global_latent"]).squeeze(-1)
        return logits, values

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        logits, values = self.forward(obs)
        _, num_ues, _ = logits.shape
        ue_mask = obs["ue_mask"]

        action_list = []
        log_prob_list = []

        for ue_idx in range(num_ues):
            dist = Categorical(logits=logits[:, ue_idx, :])

            if deterministic:
                action = torch.argmax(logits[:, ue_idx, :], dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            active_mask = ue_mask[:, ue_idx].float()

            action_list.append(action)
            log_prob_list.append(log_prob * active_mask)

        actions = torch.stack(action_list, dim=1)
        log_probs = torch.stack(log_prob_list, dim=1)
        total_log_prob = log_probs.sum(dim=1)

        return {
            "actions": actions,
            "log_prob": total_log_prob,
            "value": values,
        }

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits, values = self.forward(obs)
        _, num_ues, _ = logits.shape
        ue_mask = obs["ue_mask"]

        log_prob_list = []
        entropy_list = []

        for ue_idx in range(num_ues):
            dist = Categorical(logits=logits[:, ue_idx, :])
            action = actions[:, ue_idx]

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            active_mask = ue_mask[:, ue_idx].float()

            log_prob_list.append(log_prob * active_mask)
            entropy_list.append(entropy * active_mask)

        total_log_prob = torch.stack(log_prob_list, dim=1).sum(dim=1)
        total_entropy = torch.stack(entropy_list, dim=1).sum(dim=1)

        return {
            "log_prob": total_log_prob,
            "entropy": total_entropy,
            "value": values,
        }