from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ppo.actor_critic import Actor_Critic, ModelConfig
from ppo.buffer import RolloutBuffer
from ppo.obs_adapter import ObsAdapterConfig, ObservationAdapter
from ppo.normalizer import ObservationNormalizer, RewardNormalizer

from models import UEAction


@dataclass
class PPOTrainConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    update_epochs: int = 4
    num_updates: int = 200
    save_every: int = 20
    checkpoint_dir: str = "checkpoints"
    normalize_reward: bool = True


def to_tensor_obs(
    obs: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        x = torch.tensor(v, dtype=torch.float32, device=device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        out[k] = x
    return out


def decode_action_tensor(
    actions: torch.Tensor,
    ue_ids: list[int],
) -> Dict[int, int]:
    """
    actions: [1, N]
    """
    act = actions.squeeze(0).detach().cpu().tolist()
    return {ue_id: int(act[idx]) for idx, ue_id in enumerate(ue_ids)}


def save_checkpoint(
    model: Actor_Critic,
    optimizer: Adam,
    update: int,
    obs_normalizer: ObservationNormalizer,
    reward_normalizer: RewardNormalizer | None,
    checkpoint_dir: str,
) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    ckpt = {
        "update": update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ue_norm_mean": obs_normalizer.ue_norm.rms.mean,
        "ue_norm_var": obs_normalizer.ue_norm.rms.var,
        "ue_norm_count": obs_normalizer.ue_norm.rms.count,
        "cell_norm_mean": obs_normalizer.cell_norm.rms.mean,
        "cell_norm_var": obs_normalizer.cell_norm.rms.var,
        "cell_norm_count": obs_normalizer.cell_norm.rms.count,
    }

    if reward_normalizer is not None:
        ckpt["reward_norm_mean"] = reward_normalizer.rms.mean
        ckpt["reward_norm_var"] = reward_normalizer.rms.var
        ckpt["reward_norm_count"] = reward_normalizer.rms.count

    path = Path(checkpoint_dir) / f"ppo_update_{update:04d}.pt"
    torch.save(ckpt, path)
    print(f"[Checkpoint] saved to {path}")


def build_obs_fn(
    adapter: ObservationAdapter,
    obs_normalizer: ObservationNormalizer,
):
    def _obs_fn(state: dict) -> Dict[str, np.ndarray]:
        obs = adapter.adapt(state)
        obs = obs_normalizer.process(obs)
        return obs

    return _obs_fn


def train_one_env(env, cfg: PPOTrainConfig | None = None) -> None:
    cfg = cfg or PPOTrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) reset env lần đầu
    state, _ = env.reset()

    # 2) adapter
    adapter = ObservationAdapter(
        ObsAdapterConfig(
            n_max=16,
            m_max=16,
            include_serving_ru_feature=True,
            include_cell_type_flags=True,
        )
    )

    dims = adapter.get_feature_dims(state)
    ue_feat_dim = dims["ue_feat_dim"]
    cell_feat_dim = dims["cell_feat_dim"]

    # 3) normalizer
    obs_normalizer = ObservationNormalizer(
        ue_feat_dim=ue_feat_dim,
        cell_feat_dim=cell_feat_dim,
    )
    reward_normalizer = RewardNormalizer() if cfg.normalize_reward else None

    obs_fn = build_obs_fn(adapter, obs_normalizer)

    # 4) model
    model = Actor_Critic(
        ModelConfig(
            ue_feat_dim=ue_feat_dim,
            cell_feat_dim=cell_feat_dim,
            hidden_dim=128,
            latent_dim=128,
        )
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    for update in range(cfg.num_updates):
        buffer = RolloutBuffer()
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(cfg.rollout_steps):
            obs_np = obs_fn(state)
            obs_t = to_tensor_obs(obs_np, device)

            with torch.no_grad():
                out = model.act(obs_t, deterministic=False)

            ue_ids = sorted(state["ues"].keys())
            chosen_target = decode_action_tensor(out["actions"], ue_ids)

            actions: Dict[int, UEAction] = {}
            for ue_id in ue_ids:
                actions[ue_id] = UEAction(
                    target_ru=chosen_target[ue_id],
                    prb_alloc=0.0,
                    ptx_alloc=0.0,
                    du_alloc=0.0,
                    cu_alloc=0.0,
                )

            next_state, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            episode_reward += float(reward)

            if reward_normalizer is not None:
                reward_value = float(reward_normalizer.process(reward))
            else:
                reward_value = float(reward)

            reward_t = torch.tensor([reward_value], dtype=torch.float32, device=device)
            done_t = torch.tensor([float(done)], dtype=torch.float32, device=device)

            buffer.add(
                obs=obs_t,
                actions=out["actions"],
                log_prob=out["log_prob"],
                reward=reward_t,
                done=done_t,
                value=out["value"],
            )

            state = next_state
            if done:
                state, _ = env.reset()

        with torch.no_grad():
            obs_np = obs_fn(state)
            obs_t = to_tensor_obs(obs_np, device)
            last_value = model.act(obs_t, deterministic=True)["value"]

        batch = buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(cfg.update_epochs):
            eval_out = model.evaluate_actions(batch.obs, batch.actions)

            ratio = torch.exp(eval_out["log_prob"] - batch.old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(eval_out["value"], batch.returns)
            entropy_loss = -eval_out["entropy"].mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                + cfg.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        print(
            f"update={update:04d} "
            f"episode_reward={episode_reward:.3f} "
            f"policy_loss={policy_loss.item():.4f} "
            f"value_loss={value_loss.item():.4f}"
        )

        if (update + 1) % cfg.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                update=update + 1,
                obs_normalizer=obs_normalizer,
                reward_normalizer=reward_normalizer,
                checkpoint_dir=cfg.checkpoint_dir,
            )