from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os

from .datasets import BehaveConfig, GrabConfig, InterCapConfig, OmomoConfig, CustomDatasetConfig
from .model import ConditioningModelConfig, DenoisingModelConfig, TriDiModelConfig
from .environment import EnvironmentConfig

# ========================== RUN =============================
@dataclass
class RunConfig:
    name: str = 'debug'
    job: str = 'train'  # train, sample, eval
    cpu: bool = False
    seed: int = 42
    datasets: List[str] = field(default_factory=lambda: [])

    # Overwritten automatically during init
    path: str = os.path.join("${env.experiments_folder}", "${run.name}")


@dataclass
class ResumeConfig:
    checkpoint: Optional[str] = None  # path to the checkpoint
    step: Optional[int] = -1
    training: bool = True
    training_optimizer: bool = True
    training_scheduler: bool = True
    training_state: bool = True
# ============================================================


# ========================== TRAIN ===========================
@dataclass
class DataloaderConfig:
    batch_size: int = 8  # 2 for debug
    workers: int = 10  # 0 for debug

    sampler: str = 'weighted'  # 'default', 'random' or 'weighted'

@dataclass
class TrainConfig:
    mixed_precision: str = 'no'  # 'no', 'fp16'
    limit_train_batches: Optional[int] = None
    log_step_freq: int = 20
    checkpoint_freq: int = 50_000
    max_steps: int = 300_000
    print_step_freq: int = 100
    limit_val_batches: Optional[int] = None
    loss_t_stamp_threshold: int = 250  # threshold for loss computation

    # Losses
    losses: Dict[str, float] = field(default_factory=lambda: {
        # has to be commented out, otherwise not overwritten
        # "denoise_1": 10.0,
        # "denoise_2": 2.0,
        # "denoise_3": 5.0,
        # "smpl_v2v": 25.0,
        # "obj_v2v": 10.0,
        # "sbj_contacts": 20.0
    })

@dataclass
class OptimizerConfig:
    type: str = "torch"
    name: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-6
    scale_learning_rate_with_batch_size: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 5.0  # 5.0
    kwargs: Dict = field(default_factory=lambda: dict(betas=(0.95, 0.999)))


@dataclass
class SchedulerConfig:
    type: str = "transformers"
    name: str = "cosine"
    kwargs: Dict = field(default_factory=lambda: dict(
        num_warmup_steps=50000,  # 0
        num_training_steps="${train.max_steps}",
    ))

@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = 'tridi'
    wandb_entity: str = ''
# ============================================================

# ========================= SAMPLE ===========================
@dataclass
class SampleConfig:
    # Inference
    num_inference_steps: int = 1000
    c: Optional[str] = 'ddpm_guided'

    mode: str = "111"  # ('sbj', 'obj', 'contacts')
    contacts_mode: str = 'clip'  # 'heatmap', 'clip'

    num_samples: int = 100
    repetitions: int = 1  # for sampling for the dataset
    class_distribution: str = 'equal'  # 'equal', 'uniform'
    target: str = 'meshes'  # 'meshes', 'hdf5'
    dataset: str = 'random'  # 'random', 'normal'
    samples_file: Optional[str] = None  # overidden inside metrics
# ============================================================

# ========================== EVAL ============================
@dataclass
class EvalConfig:
    use_gen_metrics: bool = True
    use_rec_metrics: bool = True

    sampling_target: List[str] = field(default_factory=lambda: [])

# ============================================================

@dataclass
class ProjectConfig:
    behave: BehaveConfig = BehaveConfig()
    grab: GrabConfig = GrabConfig()
    intercap: InterCapConfig = InterCapConfig()
    omomo: OmomoConfig = OmomoConfig()
    custom: CustomDatasetConfig = CustomDatasetConfig()

    model_denoising: DenoisingModelConfig = DenoisingModelConfig()
    model_conditioning: ConditioningModelConfig = ConditioningModelConfig()
    model: TriDiModelConfig = TriDiModelConfig()

    env: EnvironmentConfig = EnvironmentConfig()
    run: RunConfig = RunConfig()
    resume: ResumeConfig = ResumeConfig()

    dataloader: DataloaderConfig = DataloaderConfig()
    train: TrainConfig = TrainConfig()
    logging: LoggingConfig = LoggingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    sample: SampleConfig = SampleConfig()
    eval: EvalConfig = EvalConfig()
