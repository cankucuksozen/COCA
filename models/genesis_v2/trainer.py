from dataclasses import dataclass
from typing import List

from models.base_trainer import BaseTrainer


@dataclass
class Genesis_v2_Trainer(BaseTrainer):
    @property
    def loss_terms(self) -> List[str]:
        return ["loss", "kl_loss", "recon_loss"]

    @property
    def scalar_params(self) -> List[str]:
        return ["GECO beta"]

    @property
    def param_groups(self) -> List[str]:
        return [
            "encoder",
            "seg_head",
            "icsbp",
            "feat_head",
            "z_head",
            "decoder",
            "prior_lstm",
            "prior_linear",
        ]
