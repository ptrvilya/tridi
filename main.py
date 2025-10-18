import torch

from config.config import ProjectConfig
from tridi.core.evaluator import Evaluator
from tridi.core.sampler import Sampler
from tridi.core.trainer import Trainer
from tridi.model import get_model
from tridi.utils import training as training_utils
from tridi.utils.exp import init_exp, init_wandb, init_logging, parse_arguments


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')

    # Parse arguments
    arguments = parse_arguments()

    # Initialzie run
    cfg: ProjectConfig = init_exp(arguments)

    # Logging
    init_logging(cfg)
    if cfg.logging.wandb:
        init_wandb(cfg)

    # Set random seed
    training_utils.set_seed(cfg.run.seed)

    if cfg.run.job in ['train', 'sample']:
        # Model
        model = get_model(cfg)

        if cfg.run.job == 'train':
            trainer = Trainer(cfg, model)

            trainer.train()
        elif cfg.run.job == 'sample':
            sampler = Sampler(cfg, model)
            if cfg.sample.target == 'meshes':
                sampler.sample()
            elif cfg.sample.target == 'hdf5':
                sampler.sample_to_hdf5()
            else:
                raise ValueError(f"Invalid target {cfg.sample.target}")
    elif cfg.run.job == 'eval':
        evaluator = Evaluator(cfg)
        evaluator.evaluate()
    else:
        raise ValueError(f"Invalid job type {cfg.run.job}")


if __name__ == '__main__':
    main()
