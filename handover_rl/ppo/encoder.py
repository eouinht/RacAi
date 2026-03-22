from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

@dataclass
class EncoderConfig:
    ue_feat_dim: int 
    cell_feat_dim: int
    hidden_dim: int = 128 
    latent_dim: int = 128
    
class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim:int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ObservationEncoder(nn.Module):
    
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg= cfg
        self.ue_encoder = MLPBlock(cfg.ue_feat_dim, cfg.hidden_dim, cfg.latent_dim)
        self.cell_encoder = MLPBlock(cfg.cell_feat_dim, cfg.hidden_dim, cfg.latent_dim)
        
        self.global_fusion = nn.Sequential(
            nn.Linear(cfg.latent_dim*2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.ReLU(),
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(-1)
        x = x * mask
        demon = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ue_matrix = obs["ue_matrix"]      
        cell_matrix = obs["cell_matrix"]  
        ue_mask = obs["ue_mask"]          
        cell_mask = obs["cell_mask"]      

        ue_latent = self.ue_encoder(ue_matrix)        
        cell_latent = self.cell_encoder(cell_matrix)  

        ue_global = self.masked_mean(ue_latent, ue_mask)       
        cell_global = self.masked_mean(cell_latent, cell_mask) 

        global_latent = self.global_fusion(
            torch.cat([ue_global, cell_global], dim=-1)
        ) 

        return {
            "ue_latent": ue_latent,       
            "cell_latent": cell_latent,       
            "global_latent": global_latent,   
        }