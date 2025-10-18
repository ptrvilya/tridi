"""
S_r - a test set
S_g - generated set, |S_g|=|S_r|
"""
from pathlib import Path
from typing import Union

import numpy as np

from tridi.model.nn.nn import KnnWrapper, create_nn_model
from config.config import ProjectConfig


def sample_mode_to_nn_feature(sample_mode: str):
    if "sbj_obj" in sample_mode:
        return "human_joints_object_pose"
    elif "sbj" in sample_mode:
        return "human_joints"
    elif "obj" in sample_mode:
        return "object_pose"
    else:
        raise ValueError(f"Unknown sample mode: {sample_mode}")


def coverage(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    reference_dataset: str,
    reference_set="test",  # or "train"
    sample_target="human",  # or "object"
):
    # NN:
    #   query: S_g
    #   train set: S_r
    # Initial NN setup
    # assert sample_target in ["human", "object"]
    cfg.sample.samples_file = samples_file

    # Initialize wrapper
    knn = KnnWrapper(
        model_features=sample_mode_to_nn_feature(sample_target),
        model_labels="data_source",
        model_type="general",
        backend="faiss_cpu"
    )

    # create dataset, split lists
    train_datasets = [(reference_dataset, reference_set)]
    test_datasets = [("samples", "test")]

    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    # Load data and create NN
    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    # Find NN's
    queries = test_queries['samples']
    labels = test_labels['samples']
    t_stamps = test_t_stamps['samples']
    _, pred_indices = knn.query(queries, k=1)
    # pred_labels = knn.labels[pred_indices[:, 0]]

    # Find unique indices
    unique_indices = np.unique(pred_indices)

    cov = len(unique_indices) / len(knn.labels)

    return cov


def minimum_matching_distance(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    reference_dataset: str,
    reference_set="test",  # or "train"
    sample_target="human",  # or "object"
):
    # NN:
    #   query: S_r
    #   train set: S_g
    # Initial NN setup
    # assert sample_target in ["human", "object"]
    cfg.sample.samples_file = samples_file

    # Initialize wrapper
    knn = KnnWrapper(
        model_features=sample_mode_to_nn_feature(sample_target),
        model_labels="data_source",
        model_type="general",
        backend="faiss_cpu"
    )

    # create dataset, split lists
    train_datasets = [("samples", "test")]
    test_datasets = [(reference_dataset, reference_set)]

    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    # Find NN's
    distance, counter = 0.0, 0
    # for dataset in cfg.run.datasets:
    queries = test_queries[reference_dataset]
    labels = test_labels[reference_dataset]
    t_stamps = test_t_stamps[reference_dataset]
    pred_distances, pred_indices = knn.query(queries, k=1)

    distance += np.sum(pred_distances)
    counter += len(queries)

    mmd = distance / counter
    return mmd


def nearest_neighbor_accuracy(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    reference_dataset: str,
    compare_against="test",  # or "train"
    sample_target="human",  # or "object"
    subsample: bool = False,
):
    """
        1-NNA metric
        Computed between a reference (train or test) and generated sets

    """
    # Initial NN setup
    # assert sample_target in ["human", "object"]

    datasets = [reference_dataset, "samples"]
    cfg.sample.samples_file = samples_file

    # Initialize wrapper
    knn = KnnWrapper(
        model_features=sample_mode_to_nn_feature(sample_target),
        model_labels="data_source",
        model_type="general",
        backend="faiss_cpu"
    )

    # create dataset, split lists
    train_datasets = [(dataset, compare_against) for dataset in datasets]
    test_datasets = [(dataset, compare_against) for dataset in datasets]

    # Load data and create NN
    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    # Find NN's
    generated_total, reference_total = 0, 0
    generated_hits, reference_hits = 0, 0
    for dataset in datasets:
        queries = test_queries[dataset]
        labels = test_labels[dataset]
        t_stamps = test_t_stamps[dataset]
        pred_distances, pred_indices = knn.query(queries, k=2)
        pred_labels = knn.labels[pred_indices[:, 1]]  # assuming that [0] is the query itself

        if dataset == "samples":
            generated_total += len(queries)
            generated_hits += np.sum(pred_labels == labels)
        else:
            reference_total += len(queries)
            reference_hits += np.sum(pred_labels == labels)

    # Calculate metric
    nna = (generated_hits + reference_hits) / (generated_total + reference_total)

    return nna


def sample_distance(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    reference_dataset: str,
    reference_set="train",  # or "train"
    sample_target="human",  # or "object"
):
    # NN:
    #   query: S_g
    #   train set: S_train
    # Initial NN setup
    # assert sample_target in ["human", "object"]
    cfg.sample.samples_file = samples_file

    # Initialize wrapper
    knn = KnnWrapper(
        model_features=sample_mode_to_nn_feature(sample_target),
        model_labels="data_source",
        model_type="general",
        backend="faiss_cpu"
    )

    # create dataset, split lists
    train_datasets = [(reference_dataset, reference_set)]
    test_datasets = [("samples", "test")]

    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    # Find NN's
    distance, counter = 0.0, 0
    # for dataset in cfg.run.datasets:
    queries = test_queries['samples']
    labels = test_labels['samples']
    t_stamps = test_t_stamps['samples']
    pred_distances, pred_indices = knn.query(queries, k=1)

    distance += np.sum(pred_distances)
    counter += len(queries)

    mmd = distance / counter
    return mmd