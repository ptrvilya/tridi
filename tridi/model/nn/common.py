import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def pseudo_inverse(mat):
    assert len(mat.shape) == 3
    tr = torch.bmm(mat.transpose(2, 1), mat)
    tr_inv = torch.inverse(tr)
    inv = torch.bmm(tr_inv, mat.transpose(2, 1))
    return inv


def init_object_orientation(src_axis, tgt_axis):
    pseudo = pseudo_inverse(src_axis)
    rot = torch.bmm(pseudo, tgt_axis)

    U, S, V = torch.svd(rot)
    R = torch.bmm(U, V.transpose(2, 1))

    return R


def get_sequence_from_hdf5(hdf5_file, subjects=None, actions=None, objects=None):
    sequences = []

    with h5py.File(hdf5_file, "r") as hdf5_dataset:
        for sbj in hdf5_dataset.keys():
            if subjects is not None and not(sbj in subjects):
                continue

            obj_acts = list(hdf5_dataset[sbj].keys())
            for obj_act in obj_acts:
                obj_act = str(obj_act).split("_")
                obj = obj_act[0]
                act = "_".join(obj_act[1:])

                if actions is None or (len(actions) > 0 and act in actions):
                    if objects is None or (len(objects) > 0 and obj in objects):
                        sequences.append((sbj, obj, act))

    return sequences


def get_sequences_from_split(split_file, subjects, actions, objects):
    # init output
    sequences = []

    # load split file
    with open(split_file, "r") as fp:
        split = json.load(fp)

    # form (sbj ,obj, act) triplets
    for sbj, obj_act in split:
        _oa_split = obj_act.split("_")
        obj = _oa_split[0]
        act = "_".join(_oa_split[1:])

        if subjects is None or (len(subjects) > 0 and sbj in subjects):
            if actions is None or (len(actions) > 0 and act in actions):
                if objects is None or (len(objects) > 0 and obj in objects):
                    sequences.append((sbj, obj, act))

    return sequences


def get_data_for_sequence(knn, sequence, dataset, hdf5_dataset, objname2classid):
    sbj, obj, act = sequence
    class_id = objname2classid[obj]

    hdf5_sequence = hdf5_dataset[sbj][f"{obj}_{act}"]

    T = hdf5_sequence.attrs["T"]
    T_stamps = range(T)

    labels = []

    # assert model_labels in ["data_source", "object_pose", "human_parameters"]
    #         assert model_features in ["human_joints", "human_pose", "object_pose"]
    #         assert model_type in ["class_specific", "general"]
    if knn.model_features == "human_joints":
        features = np.zeros((T, 72 * 3), dtype=np.float32)
    elif knn.model_features == "object_pose":
        features = np.zeros((T, 9 + 3), dtype=np.float32)
    elif knn.model_features == "human_parameters":
        features = np.zeros((T, 52 * 3), dtype=np.float32)
    elif knn.model_features == "human_joints_object_pose":
        features = np.zeros((T, 72 * 3 + 9 + 3), dtype=np.float32)

    if T == 0:
        return [], -1, []

    for t, t_stamp in enumerate(T_stamps):
        if knn.model_features == "human_joints":
            sbj_j = hdf5_sequence["sbj_j"][t_stamp]

            # center
            sbj_j = sbj_j - sbj_j[[0]]
            features[t] = sbj_j[1:].flatten()  # ignore root joint
        elif knn.model_features == "object_pose":
            obj_c = hdf5_sequence["obj_c"][t_stamp]
            obj_R = hdf5_sequence["obj_R"][t_stamp]

            feature = np.concatenate([obj_R.reshape(9), obj_c])
            features[t] = feature
        elif knn.model_features == "human_parameters":
            sbj_smpl_pose = hdf5_sequence["sbj_smpl_pose"][t_stamp]
            rotvec = np.zeros((52, 3), dtype=np.float32)
            for i_mat in range(sbj_smpl_pose.shape[0]):
                mat = Rotation.from_matrix(sbj_smpl_pose[i_mat])
                rotvec[i_mat] = mat.as_rotvec()
            features[t] = rotvec.flatten()
            # pose_mat = np.load(t_stamp / "subject_pose.npz")["sbj_pose"][1:]
            # pose_rotvec = np.zeros((51, 3), dtype=np.float32)
            # for i_mat in range(pose_mat.shape[0]):
            #     mat = Rotation.from_matrix(pose_mat[i_mat])
            #     pose_rotvec[i_mat] = mat.as_rotvec()
            # human_features[index] = pose_rotvec.flatten()
        elif knn.model_features == "human_joints_object_pose":
            sbj_j = hdf5_sequence["sbj_j"][t_stamp]
            # center
            sbj_j = sbj_j - sbj_j[[0]]

            obj_c = hdf5_sequence["obj_c"][t_stamp]
            obj_R = hdf5_sequence["obj_R"][t_stamp]

            obj_feature = np.concatenate([obj_R.reshape(9), obj_c])
            # ignore root joint
            features[t] = np.concatenate([
                sbj_j[1:].flatten(),
                obj_feature
            ], axis=0)
        # sbj_vertices = hdf5_sequence["sbj_v"][t_stamp]
        # scale = hdf5_sequence["prep_s"][t_stamp]
        # sbj_vertices = sbj_vertices / scale
        # human_features[t_stamp] = sbj_vertices.flatten()

        if knn.model_labels == "data_source":
            label = int(dataset == "samples")

            labels.append(label)
        elif knn.model_labels == "object_pose":
            # get object pose vectors
            # object pose is represented as Tx12 : rot matrix and center location
            obj_c = hdf5_sequence["obj_c"][t_stamp]
            obj_R = hdf5_sequence["obj_R"][t_stamp]

            label = [obj_R.reshape(9), obj_c]

            if knn.model_type == "general":
                class_label = np.array([class_id])
                label = [class_label] + label

            labels.append(np.concatenate(label))
        elif knn.model_labels == "human_parameters":
            # get human parameters
            sbj_smpl_shape = hdf5_sequence["sbj_smpl_betas"][t_stamp]
            sbj_smpl_pose = np.concatenate([
                hdf5_sequence["sbj_smpl_global"][t_stamp],
                hdf5_sequence["sbj_smpl_body"][t_stamp],
                hdf5_sequence["sbj_smpl_lh"][t_stamp],
                hdf5_sequence["sbj_smpl_rh"][t_stamp]
            ], axis=0).flatten()
            sbj_smpl_transl = hdf5_sequence["sbj_smpl_transl"][t_stamp]
            label = [sbj_smpl_shape, sbj_smpl_pose, sbj_smpl_transl]

            labels.append(np.concatenate(label))
        else:
            raise RuntimeError(f"Unknown model labels {knn.model_labels}")
    labels = np.stack(labels, axis=0)

    return features, class_id, labels


def get_hdf5_files_for_nn(cfg, dataset_list):
    hdf5_files = dict()
    for dataset_name, dataset_split in dataset_list:
        if dataset_name == "grab":
            fps = cfg.grab.fps_train if dataset_split == "train" else cfg.grab.fps_eval
            hdf5_files[dataset_name] = Path(cfg.grab.root) / f"dataset_{dataset_split}_{fps}fps.hdf5"
        elif dataset_name == "behave":
            fps = cfg.behave.fps_train if dataset_split == "train" else cfg.behave.fps_eval
            hdf5_files[dataset_name] = Path(cfg.behave.root) / f"dataset_{dataset_split}_{fps}fps.hdf5"
        elif dataset_name == "samples":
            hdf5_files[dataset_name] = Path(cfg.sample.samples_file)
        elif dataset_name == "intercap":
            fps = cfg.intercap.fps_train if dataset_split == "train" else cfg.intercap.fps_eval
            hdf5_files[dataset_name] = Path(cfg.intercap.root) / f"dataset_{dataset_split}_{fps}fps.hdf5"
        elif dataset_name == "omomo":
            fps = cfg.omomo.fps_train if dataset_split == "train" else cfg.omomo.fps_eval
            hdf5_files[dataset_name] = Path(cfg.omomo.root) / f"dataset_{dataset_split}_{fps}fps.hdf5"
    return hdf5_files


def get_sequences_for_nn(cfg, dataset_list, hdf5_files):
    sequences = dict()  # "dataset": List[(sbj, obj, act)]

    for dataset_name, dataset_split in dataset_list:
        if dataset_name == "behave":
            # save folder paths
            if dataset_split == "train":
                sequences["behave"] = get_sequences_from_split(
                    cfg.behave.train_split_file, cfg.behave.train_subjects,
                    cfg.behave.train_actions, cfg.behave.objects
                )
            else:
                sequences["behave"] = get_sequences_from_split(
                    cfg.behave.test_split_file, cfg.behave.test_subjects,
                    cfg.behave.test_actions, cfg.behave.objects
                )
        elif dataset_name == "samples":
            sequences["samples"] = get_sequence_from_hdf5(hdf5_files["samples"])
        elif dataset_name in ["grab", "intercap", "omomo"]:
            if dataset_name == "grab":
                base_config = cfg.grab
            elif dataset_name == "intercap":
                base_config = cfg.intercap
            elif dataset_name == "omomo":
                base_config = cfg.omomo

            if dataset_split == "train":
                subjects = base_config.train_subjects
            else:
                subjects = base_config.test_subjects

            sequences[dataset_name] = get_sequence_from_hdf5(
                hdf5_files[dataset_name], subjects,
                base_config.train_actions, base_config.objects
            )

    return sequences


def get_features_for_class_specific_nn(
    knn, sequences, hdf5_files, objname2classid, is_train=True
):
    # train features are concatenated
    # test features are split pre dataset
    # load training data
    if is_train:
        # {classid: data}
        features, labels, t_stamps = defaultdict(list), defaultdict(list), defaultdict(list)
    else:
        # {dataset_name: {class_id: data}}
        features, labels, t_stamps = dict(), dict(), dict()

    for dataset_name, sequences_list in sequences.items():
        if not is_train:
            features[dataset_name], labels[dataset_name], t_stamps[dataset_name] = \
                defaultdict(list), defaultdict(list), defaultdict(list)
        with h5py.File(hdf5_files[dataset_name], "r") as hdf5_dataset:
            for sequence in tqdm(sequences_list, ncols=80, leave=False):
                _features, class_id, _labels = get_data_for_sequence(
                    knn, sequence, dataset_name, hdf5_dataset, objname2classid
                )

                if len(_features) == 0:
                    continue

                sbj, obj, act = sequence
                _t_stamps = [f"{sbj}/{obj}/{act}/t{t_stamp:05d}" for t_stamp in range(len(_labels))]
                if is_train:
                    features[class_id].append(_features)
                    labels[class_id].append(_labels)
                    t_stamps[class_id].extend(_t_stamps)
                else:
                    features[dataset_name][class_id].append(_features)
                    labels[dataset_name][class_id].append(_labels)
                    t_stamps[dataset_name][class_id].extend(_t_stamps)

    for class_id in set(objname2classid.values()):
        if is_train:
            features[class_id] = np.concatenate(features[class_id], axis=0)
            labels[class_id] = np.concatenate(labels[class_id], axis=0)
            t_stamps[class_id] = t_stamps[class_id]
        else:
            for dataset_name in sequences.keys():
                if class_id in list(features[dataset_name].keys()):
                    features[dataset_name][class_id] = np.concatenate(features[dataset_name][class_id], axis=0)
                    labels[dataset_name][class_id] = np.concatenate(labels[dataset_name][class_id], axis=0)
                    t_stamps[dataset_name][class_id] = t_stamps[dataset_name][class_id]

    return features, labels, t_stamps


def get_features_for_nn(
    knn, sequences, hdf5_files, objname2classid, is_train=True
):
    # train features are concatenated
    # test features are split pre dataset
    # load training data
    if is_train:
        # [data]
        features, labels, t_stamps = [], [], []
    else:
        # {dataset_name: data}
        features, labels, t_stamps = defaultdict(list), defaultdict(list), defaultdict(list)

    for dataset_name, sequences_list in sequences.items():
        with h5py.File(hdf5_files[dataset_name], "r") as hdf5_dataset:
            for sequence in tqdm(sequences_list, ncols=80, leave=False):
                _features, _, _labels = get_data_for_sequence(
                    knn, sequence, dataset_name, hdf5_dataset, objname2classid
                )

                if len(_features) == 0:
                    continue

                sbj, obj, act = sequence
                _t_stamps = [f"{sbj}/{obj}/{act}/t{t_stamp:05d}" for t_stamp in range(len(_labels))]
                if is_train:
                    features.append(_features)
                    labels.append(_labels)
                    t_stamps.extend(_t_stamps)
                else:
                    features[dataset_name].append(_features)
                    labels[dataset_name].append(_labels)
                    t_stamps[dataset_name].extend(_t_stamps)

    # concatenate lists to arrays
    if is_train:
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        # t_stamps = t_stamps
    else:
        for dataset_name in sequences.keys():
            features[dataset_name] = np.concatenate(features[dataset_name], axis=0)
            labels[dataset_name] = np.concatenate(labels[dataset_name], axis=0)
            # t_stamps[dataset_name] = t_stamps[dataset_name]

    return features, labels, t_stamps
