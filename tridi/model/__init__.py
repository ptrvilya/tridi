from logging import getLogger

from omegaconf import OmegaConf

from config.config import ProjectConfig
from tridi.utils.training import compute_model_size
from .tridi import TriDiModel

logger = getLogger(__name__)


def get_model(cfg: ProjectConfig):
    # main model
    if cfg.model_denoising.name.endswith("unidiffuser_3"):
        model = TriDiModel(
            **OmegaConf.to_container(cfg.model, resolve=True),
            denoising_model_config=cfg.model_denoising,
            conditioning_model_config=cfg.model_conditioning
        )
    else:
        raise NotImplementedError(f"Invalid model: {cfg.model_denoising.name}")

    n_params = compute_model_size(model)
    logger.info(f"Created model: {cfg.model_denoising.name} with {n_params[0]} trainable params ({n_params[1]} total).")

    return model