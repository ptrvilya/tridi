from pathlib import Path

import torch

from tridi.data.hoi_dataset import HOIDataset
from tridi.data.random_dataset import RandomDataset
from tridi.data.batch_data import BatchData
from config.config import ProjectConfig

# logger = get_logger(__name__)
from logging import getLogger
logger = getLogger(__name__)


def get_train_dataloader(cfg: ProjectConfig):
    # list of all used datasets
    train_datasets, val_datasets = [], []

    # create datasets
    canonical_obj_meshes, canonical_obj_keypoints = dict(), dict()
    for dataset_name in cfg.run.datasets:
        if dataset_name == 'behave':
            dataset_config = cfg.behave
            train_kwargs = {
                "behave_repeat_fix": True,
                "split_file": cfg.behave.train_split_file,
            }
            val_kwargs = {
                "split_file": cfg.behave.test_split_file,
            }
        elif dataset_name == 'grab':
            dataset_config = cfg.grab
            train_kwargs, val_kwargs = dict(), dict()
        elif dataset_name == 'intercap':
            dataset_config = cfg.intercap
            train_kwargs, val_kwargs = dict(), dict()
        elif dataset_name == 'omomo':
            dataset_config = cfg.omomo
            train_kwargs, val_kwargs = dict(), dict()
        else:
            raise NotImplementedError(f'Unknown dataset: {dataset_name}')

        train_dataset = HOIDataset(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            split='train',
            objects=dataset_config.objects,
            obj2classid=dataset_config.obj2classid,
            obj2groupid=dataset_config.obj2groupid,
            downsample_factor=1,
            subjects=dataset_config.train_subjects,
            actions=dataset_config.train_actions,
            augment_rotation=dataset_config.augment_rotation if cfg.run.job == "train" else False,
            augment_symmetry=dataset_config.augment_symmetry if cfg.run.job == "train" else False,
            include_contacts=cfg.model_conditioning.use_contacts,
            include_pointnext=cfg.model_conditioning.use_pointnext_conditioning,
            assets_folder=Path(cfg.env.assets_folder),
            fps=dataset_config.fps_train,
            **train_kwargs
        )
        val_dataset = HOIDataset(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            split='test',
            objects=dataset_config.objects,
            obj2classid=dataset_config.obj2classid,
            obj2groupid=dataset_config.obj2groupid,
            downsample_factor=dataset_config.downsample_factor,
            subjects=dataset_config.test_subjects,
            actions=dataset_config.test_actions,
            include_contacts=cfg.model_conditioning.use_contacts,
            include_pointnext=cfg.model_conditioning.use_pointnext_conditioning,
            assets_folder=Path(cfg.env.assets_folder),
            fps=dataset_config.fps_eval,
            **val_kwargs
        )

        # accumulate datasets
        canonical_obj_meshes.update(train_dataset.canonical_obj_meshes)
        canonical_obj_keypoints.update(train_dataset.canonical_obj_keypoints)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # concatenate datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    if cfg.dataloader.sampler == "weighted":
        train_dataset_length = len(train_dataset)
        train_weights = []
        for dataset_name, dataset in zip(cfg.run.datasets, train_datasets):
            dataset_length = len(dataset)
            weights = torch.ones(dataset_length, dtype=torch.double)
            weights = (dataset_length / train_dataset_length) * weights

            if dataset_name == "grab":
                # weights *= 10
                weights *= 3.5
            elif dataset_name == "intercap":
                weights *= 5.5
            elif dataset_name == "omomo":
                weights *= 1.4

            train_weights.append(weights)
        train_weights = torch.cat(train_weights, dim=0)
        sampler = torch.utils.data.WeightedRandomSampler(
            train_weights, num_samples=min(100000, train_dataset_length),
            replacement=False
        )
    elif cfg.dataloader.sampler == "random":
        sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=50000)
    elif cfg.dataloader.sampler == "default":
        sampler = None
    else:
        raise NotImplementedError(f"Unknown sampler: {cfg.dataloader.sampler}")

    # create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.workers,
        drop_last=True, sampler=sampler, pin_memory=True, collate_fn=BatchData.collate,
        persistent_workers=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.workers,
        shuffle=False, pin_memory=True, collate_fn=BatchData.collate,
        persistent_workers=False,
    )
    logger.info(f"Train data length: {len(train_dataset)}")
    logger.info(f"Val data length: {len(val_dataset)}")

    return train_dataloader, val_dataloader, canonical_obj_meshes, canonical_obj_keypoints


def get_eval_dataloader(cfg: ProjectConfig):
    # list of all used datasets
    datasets = []

    # create datasets
    for dataset_name in cfg.run.datasets:
        if dataset_name == 'behave':
            dataset_config = cfg.behave
            dataset_kwargs = {
                "split_file": cfg.behave.test_split_file,
            }
        elif dataset_name == 'grab':
            dataset_config = cfg.grab
            dataset_kwargs = dict()
        elif dataset_name == 'intercap':
            dataset_config = cfg.intercap
            dataset_kwargs = dict()
        elif dataset_name == 'omomo':
            dataset_config = cfg.omomo
            dataset_kwargs = dict()
        elif dataset_name == 'custom':
            dataset_config = cfg.custom
            dataset_kwargs = {
                "split_file": cfg.custom.test_split_file,
            }
        else:
            raise NotImplementedError(f'Unknown dataset: {dataset_name}')

        dataset = HOIDataset(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            split='test',
            objects=dataset_config.objects,
            obj2classid=dataset_config.obj2classid,
            obj2groupid=dataset_config.obj2groupid,
            downsample_factor=1,
            subjects=dataset_config.test_subjects,
            actions=dataset_config.test_actions,
            include_contacts=cfg.model_conditioning.use_contacts,
            include_pointnext=cfg.model_conditioning.use_pointnext_conditioning,
            assets_folder=Path(cfg.env.assets_folder),
            fps=dataset_config.fps_eval,
            **dataset_kwargs
        )

        # accumulate datasets
        datasets.append(dataset)

    # create dataloaders
    dataloaders, canonical_obj_meshes, canonical_obj_keypoints = [], dict(), dict()
    for dataset in datasets:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.workers,
            shuffle=False, pin_memory=True, collate_fn=BatchData.collate
        )

        canonical_obj_meshes.update(dataset.canonical_obj_meshes)
        canonical_obj_keypoints.update(dataset.canonical_obj_keypoints)
        dataloaders.append(dataloader)
        logger.info(f"Eval data length for {dataset.name}: {len(dataset)}")

    return dataloaders, canonical_obj_meshes, canonical_obj_keypoints


def get_eval_dataloader_random(cfg: ProjectConfig):
    # list of all used datasets
    datasets = []

    # create datasets
    for dataset_name in cfg.run.datasets:
        if dataset_name == 'behave':
            dataset_config = cfg.behave
            dataset_kwargs = {
                "split_file": cfg.behave.test_split_file,
            }
        elif dataset_name == 'grab':
            dataset_config = cfg.grab
            dataset_kwargs = dict()
        elif dataset_name == 'intercap':
            dataset_config = cfg.intercap
            dataset_kwargs = dict()
        elif dataset_name == 'omomo':
            dataset_config = cfg.omomo
            dataset_kwargs = dict()
        else:
            raise NotImplementedError(f'Unknown dataset: {dataset_name}')

        dataset = RandomDataset(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            objects=dataset_config.objects,
            obj2classid=dataset_config.obj2classid,
            obj2groupid=dataset_config.obj2groupid,
            num_samples=cfg.sample.num_samples,
            class_distribution=cfg.sample.class_distribution,
            include_contacts=cfg.model_conditioning.use_contacts,
            include_pointnext=cfg.model_conditioning.use_pointnext_conditioning,
            assets_folder=Path(cfg.env.assets_folder),
        )

        # accumulate datasets
        datasets.append(dataset)

    # create dataloaders
    dataloaders, canonical_obj_meshes, canonical_obj_keypoints = [], dict(), dict()
    for dataset in datasets:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.dataloader.batch_size, num_workers=cfg.dataloader.workers,
            shuffle=False, pin_memory=True, collate_fn=BatchData.collate,
        )

        canonical_obj_meshes.update(dataset.canonical_obj_meshes)
        canonical_obj_keypoints.update(dataset.canonical_obj_keypoints)
        dataloaders.append(dataloader)
        logger.info(f"Random data length for {dataset.name}: {len(dataset)}")

    return dataloaders, canonical_obj_meshes, canonical_obj_keypoints
