"""
Implementation of nearest neighbours baseline for Object pop-up with and without class prediction.
"""
from typing import Union, Dict

import numpy as np
try:
    import faiss
except ImportError:
    pass
from scipy.spatial import KDTree

from config.config import ProjectConfig
from tridi.model.nn.common import (
    get_hdf5_files_for_nn, get_sequences_for_nn,
    get_features_for_class_specific_nn, get_features_for_nn
)


class KnnWrapper:
    def __init__(
        self,
        model_features="human", model_labels="object",
        model_type="human_general", backend="faiss_cpu"
    ):
        assert model_features in [
            "human_joints", "human_pose", "object_pose", "human_joints_object_pose"
        ]
        assert model_labels in ["data_source", "object_pose", "human_parameters"]
        assert model_type in ["class_specific", "general"]

        self.backend = backend
        self.model_type = model_type
        self.model_features = model_features
        self.model_labels = model_labels

        self.index = None
        self.data = None
        self.labels = None

    def create_index(
        self,
        features: Union[np.ndarray, Dict[int, np.ndarray]],
        labels: Union[np.ndarray, Dict[int, np.ndarray]]
    ):
        if self.model_type == 'general':
            if self.backend == 'scipy':
                self.index = KDTree(features, copy_data=True)
            elif self.backend == 'faiss_cpu':
                self.index = faiss.IndexFlatL2(features.shape[1])
                self.index.add(features.astype(np.float32))
            else:
                resources = faiss.StandardGpuResources()
                flat_config = faiss.GpuIndexFlatConfig()
                flat_config.device = 0

                self.index = faiss.GpuIndexFlatL2(resources, features.shape[1], flat_config)
                self.index.add(features.astype(np.float32))
        else:
            self.index = dict()

            for class_id in features.keys():
                if self.backend == 'scipy':
                    self.index[class_id] = KDTree(features[class_id], copy_data=True)
                elif self.backend == 'faiss_cpu':
                    self.index[class_id] = faiss.IndexFlatL2(features[class_id].shape[1])
                    self.index[class_id].add(features[class_id].astype(np.float32))
                else:
                    resources = faiss.StandardGpuResources()
                    flat_config = faiss.GpuIndexFlatConfig()
                    flat_config.device = 0

                    self.index[class_id] = faiss.GpuIndexFlatL2(resources, features[class_id].shape[1], flat_config)
                    self.index[class_id].add(features[class_id].astype(np.float32))
        self.data = features
        self.labels = labels

    def query(self, features, k=1, class_id=None):
        if self.model_type == 'class_specific':
            assert class_id is not None
        knn = self.index if self.model_type == 'general' else self.index[class_id]

        if self.backend == "scipy":
            distances, indices = knn.query(features.astype(np.float32), k=k)
        else:
            distances, indices = knn.search(features.astype(np.float32), k=k)

        return distances, indices


def create_nn_model(
    cfg: ProjectConfig,
    knn: KnnWrapper,
    train_datasets: list,
    test_datasets: list,
):
    # Default class mapping
    objname2classid = dict()
    for dataset_name, _ in train_datasets + test_datasets:
        if dataset_name == "grab":
            objname2classid.update(cfg.grab.obj2classid)
        elif dataset_name == "behave":
            objname2classid.update(cfg.behave.obj2classid)
        elif dataset_name == "intercap":
            objname2classid.update(cfg.intercap.obj2classid)
        elif dataset_name == "omomo":
            objname2classid.update(cfg.omomo.obj2classid)

    # ===> 1. Locate hdf5 files
    train_hdf5 = get_hdf5_files_for_nn(cfg, train_datasets)
    test_hdf5 = get_hdf5_files_for_nn(cfg, test_datasets)
    # <===

    # ===> 2. Get train / test sequences
    # "dataset": List[(sbj, obj, act)]
    train_sequences = get_sequences_for_nn(cfg, train_datasets, train_hdf5)
    test_sequences = get_sequences_for_nn(cfg, test_datasets, test_hdf5)

    # <===

    # ===> 3. Load features
    if knn.model_type == 'general':
        train_features, train_labels, train_t_stamps = get_features_for_nn(
            knn, train_sequences, train_hdf5, objname2classid, is_train=True
        )
        test_queries, test_labels, test_t_stamps = get_features_for_nn(
            knn, test_sequences, test_hdf5, objname2classid, is_train=False
        )
    else:
        train_features, train_labels, train_t_stamps = get_features_for_class_specific_nn(
            knn, train_sequences, train_hdf5, objname2classid, is_train=True
        )
        test_queries, test_labels, test_t_stamps = get_features_for_class_specific_nn(
            knn, test_sequences, test_hdf5, objname2classid, is_train=False
        )

    # build kdtree on training data
    knn.create_index(train_features, train_labels)

    return knn, test_queries, test_labels, test_t_stamps


def create_nn_baseline(
    cfg: ProjectConfig,
    model_type: str = 'class_specific',  # or 'general'
    sample_target: str = "human",  # or "object"
    backend: str = 'faiss_gpu'
):
    # Initialize wrapper
    knn = KnnWrapper(
        model_features="object_pose" if sample_target == 'human' else "human_joints",
        model_labels="human_parameters" if sample_target == 'human' else "object_pose",
        model_type=model_type,
        backend=backend
    )

    # create dataset, split lists
    train_datasets = [(dataset, "train") for dataset in cfg.run.datasets]
    test_datasets = [(dataset, "test") for dataset in cfg.run.datasets]

    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    return knn, test_queries, test_labels, test_t_stamps


# if dataset == "grab":
#     pass
#     data_paths["grab"] = cfg.grab_path
#
#     # get train/test sequences
#     train_subjects, train_objects = cfg.grab["train_subjects"], cfg.grab["train_objects"]
#     val_subjects, val_objects = cfg.grab["val_subjects"], cfg.grab["val_objects"]
#
#     # get (sbj, obj_act) tuples for training
#     train_objact = get_sbj_to_obj_act(
#         path_template, data_paths["grab"], train_subjects, train_objects, actions=None
#     )
#     # form folder path for each pair
#     grab_train_folders = []
#     for subject, obj_acts in train_objact.items():
#         for (obj, act) in obj_acts:
#             grab_train_folders.append(path_template.format(subject=subject, object=obj, action=act))
#
#     # get (sbj, obj_act) tuples for testing
#     val_objact = get_sbj_to_obj_act(
#         path_template, data_paths["grab"], val_subjects, val_objects, actions=None
#     )
#     # form folder path for each pair
#     grab_val_folders = []
#     for subject, obj_acts in val_objact.items():
#         for (obj, act) in obj_acts:
#             grab_val_folders.append(path_template.format(subject=subject, object=obj, action=act))
#     # save folder paths
#     train_folders["grab"] = grab_train_folders
#     val_folders["grab"] = grab_val_folders


# def create_and_query_nn_model(cfg, model_type: str, n_neighbors=1, human_features="verts", backend="scipy"):
#     # function that creates and quires the model simultaneously to optimize compute and storage
#     assert model_type in ["pose_class_specific"]  # pose_class_specific - (class, vertices) -> object_pose
#
#     # Default class mapping
#     objname2classid = cfg.objname2classid
#
#     # ===> 1. Get folders with data
#     train_folders, val_folders = dict(), dict()
#     data_paths = {}
#     for dataset in cfg.datasets:
#         path_template = "{subject}/{object}_{action}"
#         if dataset == "grab":
#             data_paths["grab"] = cfg.grab_path
#
#             # get train/test sequences
#             train_subjects, train_objects = cfg.grab["train_subjects"], cfg.grab["train_objects"]
#             val_subjects, val_objects = cfg.grab["val_subjects"], cfg.grab["val_objects"]
#
#             # get (sbj, obj_act) tuples for training
#             train_objact = get_sbj_to_obj_act(
#                 path_template, data_paths["grab"], train_subjects, train_objects, actions=None
#             )
#             # form folder path for each pair
#             grab_train_folders = []
#             for subject, obj_acts in train_objact.items():
#                 for (obj, act) in obj_acts:
#                     grab_train_folders.append(path_template.format(subject=subject, object=obj, action=act))
#
#             # get (sbj, obj_act) tuples for testing
#             val_objact = get_sbj_to_obj_act(
#                 path_template, data_paths["grab"], val_subjects, val_objects, actions=None
#             )
#             # form folder path for each pair
#             grab_val_folders = []
#             for subject, obj_acts in val_objact.items():
#                 for (obj, act) in obj_acts:
#                     grab_val_folders.append(path_template.format(subject=subject, object=obj, action=act))
#             # save folder paths
#             train_folders["grab"] = grab_train_folders
#             val_folders["grab"] = grab_val_folders
#         elif dataset == "behave":
#             data_paths["behave"] = cfg.behave_path
#
#             # get train/test sequences
#             train_split, train_objects = cfg.behave["train_split_file"], cfg.behave["train_objects"]
#             val_split, val_objects = cfg.behave["val_split_file"], cfg.behave["val_objects"]
#
#             # get (sbj, obj_act) tuples for training
#             with open(train_split, "r") as fp:
#                 split = json.load(fp)
#             # form folder path for each pair
#             behave_train_folders = []
#             for subject, obj_act in split:
#                 obj_act = obj_act.split("_")
#                 obj = obj_act[0]
#                 act = "_".join(obj_act[1:])
#                 if obj in train_objects:
#                     behave_train_folders.append(path_template.format(subject=subject, object=obj, action=act))
#
#             # get (sbj, obj_act) tuples for testing
#             with open(val_split, "r") as fp:
#                 split = json.load(fp)
#             # form folder path for each pair
#             behave_val_folders = []
#             for subject, obj_act in split:
#                 obj_act = obj_act.split("_")
#                 obj = obj_act[0]
#                 act = "_".join(obj_act[1:])
#                 if obj in val_objects:
#                     behave_val_folders.append(path_template.format(subject=subject, object=obj, action=act))
#
#             # save folder paths
#             train_folders["behave"] = behave_train_folders
#             val_folders["behave"] = behave_val_folders
#     # <===
#
#     # ===> 2. Load data
#     # load training data
#     train_features, _train_labels = defaultdict(list), defaultdict(list)
#     for dataset, folders in train_folders.items():
#         for folder in tqdm(folders, ncols=80):
#             features, class_id, labels = get_data_for_sequence(
#                 human_features, model_type, data_paths[dataset], folder, objname2classid
#             )
#
#             if len(features) == 0:
#                 continue
#
#             train_features[class_id].append(features)
#             _train_labels[class_id].append(labels)
#
#     # load test data
#     _test_queries, _test_labels, test_t_stamps = dict(), dict(), dict()
#     for dataset, folders in val_folders.items():
#         _test_queries[dataset], _test_labels[dataset], test_t_stamps[dataset] = \
#             defaultdict(list), defaultdict(list), defaultdict(list)
#
#         for folder in folders:
#             features, class_id, labels = get_data_for_sequence(
#                 human_features, model_type, data_paths[dataset], folder, objname2classid
#             )
#
#             if len(features) == 0:
#                 continue
#
#             _test_queries[dataset][class_id].append(features)
#             _test_labels[dataset][class_id].append(labels)
#             test_t_stamps[dataset][class_id].extend(
#                 [f"{folder}/t{t_stamp:05d}" for t_stamp in range(len(labels))]
#             )
#
#     test_queries, test_labels = dict(), dict()
#     for dataset in val_folders.keys():
#         test_queries[dataset], test_labels[dataset] = dict(), dict()
#         for class_id in _test_queries[dataset].keys():
#             test_queries[dataset][class_id] = np.concatenate(_test_queries[dataset][class_id], axis=0)
#             test_labels[dataset][class_id] = np.concatenate(_test_labels[dataset][class_id], axis=0)
#     # <===
#
#     # ===> 3. Create and query models
#     # build and query per-class kdtrees
#     pred_neighbors, train_labels = dict(), dict()
#     for class_id in train_features.keys():
#         pred_neighbors[class_id] = dict()
#         features = np.concatenate(train_features[class_id], axis=0).astype(np.float32)
#         train_labels[class_id] = np.concatenate(_train_labels[class_id], axis=0)
#
#         if backend == "scipy":
#             kdtree = KDTree(features, copy_data=True)
#         elif backend == "faiss_cpu":
#             kdtree = KnnFaiss(features, device="cpu")
#         elif backend == "faiss_gpu":
#             kdtree = KnnFaiss(features, device="gpu")
#
#         for dataset in test_queries.keys():
#             if class_id in test_queries[dataset]:
#                 test_query = test_queries[dataset][class_id]
#                 _, pred_neighbors[class_id][dataset] = kdtree.query(test_query, k=n_neighbors)
#     # <===
#
#     return pred_neighbors, train_labels, test_labels, test_t_stamps