from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os


# ========================== MODEL ===========================
@dataclass
class ConditioningModelConfig:
    # class number
    use_class_conditioning: bool = True
    num_classes: int = 40  # actually number of groups - 40, 52, 64
    # pointnext encoding
    use_pointnext_conditioning: bool = True
    # contacts - for 3way unidiffuser
    use_contacts: str = "encoder_decimated_clip"  # "surface", "parts"
    contact_model: str = "gb_contacts"


@dataclass
class DenoisingModelConfig:
    name: str = "transformer" "_" + "unidiffuser_3"  # 'simple','transformer' x "joint", "unidiffuser"
    dim_timestep_embed: int = 128
    params: Dict = field(default_factory=lambda: {})


@dataclass
class TriDiModelConfig:
    # Input configuration
    data_sbj_channels: int = 10 + 52 * 6 + 3
    data_obj_channels: int = 3 + 6
    data_contact_channels: int = 128  # 256 for surface, 24 for parts

    # diffusion
    denoise_mode: str = 'sample'  # epsilon or sample (as in scheduler - prediction_type)
    beta_start: float = 1e-5  # 0.00085
    beta_end: float = 8e-3  # 0.012
    beta_schedule: str = 'linear'  # 'custom'

    # guidance for the model
    cg_apply: bool = False
    cg_scale: float = 0.0
    cg_t_stamp: int = 200
# ============================================================