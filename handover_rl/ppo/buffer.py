from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class RolloutBatch:
    obs: Dict[str, torch.Tensor]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.obs_list: List[Dict[str, torch.Tensor]] = []
        self.actions_list: List[torch.Tensor] = []
        self.log_probs_list: List[torch.Tensor] = []
        self.rewards_list: List[torch.Tensor] = []
        self.dones_list: List[torch.Tensor] = []
        self.values_list: List[torch.Tensor] = []

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.obs_list.append({k: v.detach().clone() for k, v in obs.items()})
        self.actions_list.append(actions.detach().clone())
        self.log_probs_list.append(log_prob.detach().clone())
        self.rewards_list.append(reward.detach().clone())
        self.dones_list.append(done.detach().clone())
        self.values_list.append(value.detach().clone())

    def clear(self) -> None:
        self.__init__()

    def _stack_obs(self) -> Dict[str, torch.Tensor]:
        keys = self.obs_list[0].keys()
        return {k: torch.cat([obs[k] for obs in self.obs_list], dim=0) for k in keys}

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        rewards = self.rewards_list
        dones = self.dones_list
        values = self.values_list + [last_value.detach()]

        advantages = []
        gae = torch.zeros_like(last_value)

        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)

        advantages_t = torch.cat(advantages, dim=0)
        values_t = torch.cat(self.values_list, dim=0)
        returns_t = advantages_t + values_t

        return RolloutBatch(
            obs=self._stack_obs(),
            actions=torch.cat(self.actions_list, dim=0),
            old_log_probs=torch.cat(self.log_probs_list, dim=0),
            returns=returns_t,
            advantages=advantages_t,
            values=values_t,
        )