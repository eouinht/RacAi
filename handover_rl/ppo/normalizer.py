from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RunningStatConfig:
    epsilon: float = 1e-8
    clip_value: float = 10.0


class RunningMeanStd:
    """
    Running mean/std theo kiểu online update.
    Dùng cho feature normalization hoặc reward normalization.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """
        x shape: [B, ...]
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[None, ...]

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


class FeatureNormalizer:
    """
    Chuẩn hóa feature matrix với running statistics.
    Hỗ trợ:
    - fit/update online
    - normalize
    - freeze khi eval
    """

    def __init__(
        self,
        feature_dim: int,
        epsilon: float = 1e-8,
        clip_value: float = 10.0,
    ) -> None:
        self.rms = RunningMeanStd(shape=(feature_dim,))
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.training = True

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def update(self, x: np.ndarray, mask: np.ndarray | None = None) -> None:
        """
        x: [K, F] hoặc [B, K, F]
        mask: [K] hoặc [B, K]
        Chỉ update trên phần tử mask=1 nếu có mask.
        """
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 2:
            x_flat = x
            if mask is not None:
                mask = np.asarray(mask, dtype=np.float32)
                valid = mask > 0
                x_flat = x_flat[valid]
        elif x.ndim == 3:
            b, k, f = x.shape
            x_flat = x.reshape(b * k, f)
            if mask is not None:
                mask = np.asarray(mask, dtype=np.float32).reshape(b * k)
                valid = mask > 0
                x_flat = x_flat[valid]
        else:
            raise ValueError(f"Unsupported x ndim={x.ndim}")

        if x_flat.size == 0:
            return

        self.rms.update(x_flat)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = (x - self.rms.mean.astype(np.float32)) / (
            self.rms.std.astype(np.float32) + self.epsilon
        )
        out = np.clip(out, -self.clip_value, self.clip_value)
        return out.astype(np.float32)

    def process(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if self.training:
            self.update(x, mask)
        return self.normalize(x)


class RewardNormalizer:
    """
    Chuẩn hóa reward scalar bằng running std.
    Thường dùng:
    - normalize reward online
    - hoặc normalize return target
    """

    def __init__(self, epsilon: float = 1e-8, clip_value: float = 10.0) -> None:
        self.rms = RunningMeanStd(shape=())
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.training = True

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def process(self, reward: float | np.ndarray) -> np.ndarray:
        r = np.asarray(reward, dtype=np.float32)
        if self.training:
            self.rms.update(r.reshape(-1))
        out = r / (float(self.rms.std) + self.epsilon)
        out = np.clip(out, -self.clip_value, self.clip_value)
        return out.astype(np.float32)


class ObservationNormalizer:
    """
    Normalizer tổng cho observation:
    - ue_matrix
    - cell_matrix
    Reward để riêng.
    """

    def __init__(
        self,
        ue_feat_dim: int,
        cell_feat_dim: int,
        epsilon: float = 1e-8,
        clip_value: float = 10.0,
    ) -> None:
        self.ue_norm = FeatureNormalizer(
            feature_dim=ue_feat_dim,
            epsilon=epsilon,
            clip_value=clip_value,
        )
        self.cell_norm = FeatureNormalizer(
            feature_dim=cell_feat_dim,
            epsilon=epsilon,
            clip_value=clip_value,
        )

    def train(self) -> None:
        self.ue_norm.train()
        self.cell_norm.train()

    def eval(self) -> None:
        self.ue_norm.eval()
        self.cell_norm.eval()

    def process(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        ue_matrix = obs["ue_matrix"]
        cell_matrix = obs["cell_matrix"]
        ue_mask = obs["ue_mask"]
        cell_mask = obs["cell_mask"]

        ue_matrix_norm = self.ue_norm.process(ue_matrix, ue_mask)
        cell_matrix_norm = self.cell_norm.process(cell_matrix, cell_mask)

        return {
            "ue_matrix": ue_matrix_norm.astype(np.float32),
            "cell_matrix": cell_matrix_norm.astype(np.float32),
            "ue_mask": ue_mask.astype(np.float32),
            "cell_mask": cell_mask.astype(np.float32),
        }