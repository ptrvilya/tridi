"""
Code to preprocess 1fps annotations for the BEAHVE dataset.
"""
import argparse
import json
import pickle as pkl
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import compress
from multiprocessing import set_start_method
from pathlib import Path

import h5py
import numpy as np
import smplx
import torch
import tqdm
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from .common import tensor_to_cpu, estimate_transform, DatasetSample, preprocess_worker, contacts_worker, \
    get_sequences_list, th_posemap_axisang, generate_obj_keypoints_from_barycentric, \
    trimesh_load, generate_behave_canonicalized_objects, add_sequence_datasets_to_hdf5, add_meatada_to_hdf5, \
    init_preprocessing, generate_object_meshes
from ..utils.parallel_map import parallel_map


def preprocess(cfg):
    set_start_method('spawn')

    # convert to Path
    root_folder = Path(cfg.behave.root)
    target_folder = Path(cfg.behave.target)
    can_objects_path = Path(cfg.behave.can_objects_path)

    # list dataset sequences
    _sequences = get_sequences_list(
        "behave", root_folder / "sequences", objects=cfg.behave.objects, subjects=cfg.behave.subjects
    )

    # filter sequences based on split
    if cfg.behave.split in ["train", "test"]:
        with open(cfg.behave.split_file, "r") as fp:
            split = json.load(fp)
        split_sequences = split[cfg.behave.split]
        sequences = [seq for seq in _sequences if seq.name in split_sequences]
        hdf5_name = f"dataset_{cfg.behave.split}"
    else:
        sequences = _sequences
        hdf5_name = "dataset"
    # useful to combine with already created annotations
    if cfg.behave.combine_1fps_and_30fps:
        hdf5_name += f"_{cfg.behave.downsample}"
        mode = 'a'
    else:
        hdf5_name += "_1fps"
        mode ='w'

    # init hdf5 file
    subjects = list(set([s.name.split("_")[1] for s in sequences]))
    target_folder.mkdir(exist_ok=True, parents=True)

    h5py_file = h5py.File(str(target_folder / f"{hdf5_name}.hdf5"), mode)
    for sbj in subjects:
        group_name = \
            f"{sbj}_{cfg.behave.split}" if cfg.behave.split in ["train", "test"] else sbj
        if not group_name in h5py_file:
            h5py_file.create_group(group_name)

    # preprocess each sequence
    contact_masks = {}
    for sequence in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        # load sequence info
        with (sequence / "info.json").open("r") as fp:
            sequence_info = json.load(fp)  # 'cat', 'gender'

        # parse sequence name: Date<:02d>_Sub<:02d>_<object>_<optional:action>
        sequence_name = sequence.name.split("_")
        seq_date = sequence_name[0]
        seq_subject = sequence_name[1]
        seq_action = f"{sequence_name[3]}_{seq_date}" if len(sequence_name) == 4 else f"{seq_date}"
        seq_object = sequence_info["cat"]

        # Dataset structure
        # person/fit02/person_fit.pkl: ['pose', 'betas', 'trans', 'score']
        #   shapes: 156, 10, 3, 0
        # <object>/fit01/<object>_fit.pkl: ['angle', 'trans']
        #   shapes: 3, 3

        # ============ 1 extract vertices for subject
        t_stamps = sorted(list(sequence.glob("t????.000")))
        T = len(t_stamps)
        preprocess_transforms = []
        if cfg.input_type in ["smplh", "smpl"]:
            # load sbj mesh to use as a template
            sbj_mesh = trimesh_load(t_stamps[0] / "person/fit02/person_fit.ply")

            # create smplh model
            sbj_model = smplx.build_layer(
                model_path=str(cfg.env.smpl_folder), model_type="smplh", gender=sequence_info["gender"],
                use_pca=False, num_betas=10, batch_size=T,
            )

            # load smpl(-h) parameters
            smpl_params = defaultdict(list)
            for t_stamp in t_stamps:
                with (t_stamp / "person/fit02/person_fit.pkl").open("rb") as fp:
                    model_params = pkl.load(fp)
                smpl_params["betas"].append(model_params["betas"])
                smpl_params["pose"].append(model_params["pose"])
                smpl_params["trans"].append(model_params["trans"])
            smpl_params = {k: np.array(v, dtype=np.float32) for k, v in smpl_params.items()}

            # convert parameters
            th_pose_axisangle = torch.tensor(smpl_params["pose"]).reshape(T, 52, 3)
            th_pose_rotmat = th_posemap_axisang(th_pose_axisangle.reshape(T * 52, 3)).reshape(T, 52, 9)
            body_model_params = {
                "betas": torch.tensor(smpl_params['betas']),
                "transl": torch.tensor(smpl_params["trans"]),
                "global_orient": th_pose_rotmat[:, :1].reshape(T, -1, 9),
                "body_pose": th_pose_rotmat[:, 1:22].reshape(T, -1, 9),
                "left_hand_pose": th_pose_rotmat[:, 22:37].reshape(T, -1, 9),
                "right_hand_pose": th_pose_rotmat[:, 37:].reshape(T, -1, 9),
            }
            if cfg.input_type == "smpl":
                body_model_params["left_hand_pose"] = None
                body_model_params["right_hand_pose"] = None

            # get smpl(-h) vertices
            sbj_output = sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
            sbj_verts = tensor_to_cpu(sbj_output.vertices)
            sbj_joints = tensor_to_cpu(sbj_output.joints)
            sbj_transl = body_model_params["transl"].numpy()
            sbj_orient = body_model_params["global_orient"].numpy().reshape(T, 3, 3)

            # align sbj meshes
            raw_pc_vertices = []
            for i in range(T):
                # custom rotation to align with grab data
                R_grab = Rotation.from_euler('x', [-90], degrees=True)

                if cfg.align_with_joints:
                    old_transl = sbj_transl[i]
                    pelvis = sbj_joints[i][0] - old_transl
                    rot_center = pelvis + old_transl

                    # remove effect of global transl and center in pelvis
                    sbj_joints[i] = sbj_joints[i] - rot_center
                    sbj_verts[i] = sbj_verts[i] - rot_center

                    # rotate around pelvis
                    sbj_joints[i] = R_grab.apply(sbj_joints[i])

                    # rotation to align torso with x-axis direction
                    shoulderblades = sbj_joints[i][2] - sbj_joints[i][1]
                    z = np.array([0, 0, 1])
                    dir_shoulderblades = np.cross(z, shoulderblades)
                    dir_shoulderblades[2] = 0.0  # project to xy
                    dir_shoulderblades = dir_shoulderblades / np.linalg.norm(dir_shoulderblades)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        R_align, _ = Rotation.align_vectors(np.array([[1, 0, 0]]), dir_shoulderblades[None])
                    # q followed by p is equivalent to p * q.
                    R = R_align * R_grab

                    # rotate around pelvis
                    sbj_verts[i] = R.apply(sbj_verts[i])
                    sbj_joints[i] = R_align.apply(sbj_joints[i])

                    # reapply pelvis and global translation
                    sbj_joints[i] = sbj_joints[i] + rot_center
                    sbj_verts[i] = sbj_verts[i] + rot_center

                    t_reset = -1 * np.copy(sbj_joints[i][0])  # center using root joint
                    sbj_joints[i] += t_reset
                    sbj_verts[i] += t_reset

                    # new global smpl params
                    sbj_orient[i] = (R * Rotation.from_matrix(sbj_orient[i])).as_matrix()
                    sbj_transl[i] += t_reset
                else:
                    # only align using center
                    # R_align = Rotation.from_matrix(np.eye(3, dtype=np.float32))
                    old_transl = sbj_transl[i]
                    pelvis = sbj_joints[i][0] - old_transl
                    rot_center = pelvis + old_transl

                    # remove effect of global transl and center in pelvis
                    sbj_joints[i] = sbj_joints[i] - rot_center
                    sbj_verts[i] = sbj_verts[i] - rot_center

                    # rotate around pelvis
                    R = R_grab
                    sbj_verts[i] = R.apply(sbj_verts[i])
                    sbj_joints[i] = R.apply(sbj_joints[i])

                    # reapply pelvis and global translation
                    sbj_joints[i] = sbj_joints[i] + rot_center
                    sbj_verts[i] = sbj_verts[i] + rot_center

                    # t_reset = -1 * np.mean(sbj_verts[i], axis=0)
                    t_reset = -1 * sbj_joints[i][0]
                    sbj_joints[i] += t_reset
                    sbj_verts[i] += t_reset

                    # new global smpl params
                    sbj_orient[i] = (R * Rotation.from_matrix(sbj_orient[i])).as_matrix()
                    sbj_transl[i] += t_reset

                preprocess_transforms.append({
                    "R": R.as_matrix(),
                    "t": np.copy(t_reset),
                    "rot_center": np.copy(rot_center)
                })

                if cfg.behave.use_raw_pcs:
                    raw_pc = np.array(trimesh_load(t_stamps[i] / "person/person.ply").vertices)
                    raw_pc = R.apply(raw_pc - rot_center) + rot_center + t_reset
                    raw_pc_vertices.append(raw_pc)

            # save smpl parameters
            sbj_smpl = {
                "betas": body_model_params["betas"],
                "transl": sbj_transl,
                "global_orient": sbj_orient.reshape(T, 1, 9),
                "body_pose": body_model_params["body_pose"].reshape(T, -1, 9).numpy(),
                "left_hand_pose": body_model_params["left_hand_pose"].reshape(T, -1, 9).numpy(),
                "right_hand_pose": body_model_params["right_hand_pose"].reshape(T, -1, 9).numpy()
            }

            sbj_verts = np.stack(sbj_verts, axis=0)
            sbj_joints = np.stack(sbj_joints, axis=0)
        else:
            raise ValueError(f"Unsupported input data type {cfg.input_type}")
        # ===========================================

        # ============ 2 extract vertices for object
        # mapping from class names to saved mesh names
        if seq_object in ["chairblack", "chairwood"]:
            _object_name = "chair"
        elif seq_object == "basketball":
            _object_name = "sports ball"
        elif seq_object == "yogaball":
            _object_name = "sports ball"
        else:
            _object_name = seq_object

        # load object mesh and transform using sbj transformation
        obj_verts = []
        for index, t_stamp in enumerate(t_stamps):
            obj_mesh = trimesh_load(t_stamp / f"{_object_name}/fit01/{_object_name}_fit.ply")

            obj_mesh_v = np.array(obj_mesh.vertices)
            R = Rotation.from_matrix(preprocess_transforms[index]["R"])
            t = preprocess_transforms[index]["t"]
            rot_center = preprocess_transforms[index]["rot_center"]
            obj_mesh.vertices = R.apply(obj_mesh_v - rot_center) + rot_center + t
            obj_verts.append(np.array(obj_mesh.vertices))
        obj_verts = np.stack(obj_verts, axis=0)
        # ===========================================

        # ============ 3 filter based on contacts
        # load canonicalized object mesh
        obj_mesh = trimesh_load(can_objects_path / f"{seq_object}.ply")

        input_data = [{
            "obj_mesh": deepcopy(obj_mesh),
            "obj_verts": obj_verts[t],
            "sbj_verts": sbj_verts[t],
            "sbj_faces": sbj_mesh.faces,
            "contact_threshold": cfg.behave.contact_threshold
        } for t in range(T)]

        # Calculate contacts
        contact_mask = parallel_map(
            input_data, contacts_worker, use_kwargs=True, n_jobs=24, tqdm_kwargs={"leave": False}
        )
        contact_mask = np.array(contact_mask)
        T = contact_mask.sum()

        # if no frame is selected continue to the next sequence
        if T < 1:
            continue

        sbj_verts = sbj_verts[contact_mask]
        obj_verts = obj_verts[contact_mask]
        sbj_joints = sbj_joints[contact_mask]
        sbj_smpl = {k: v[contact_mask] for k, v in sbj_smpl.items()}
        preprocess_transforms = list(compress(preprocess_transforms, contact_mask))
        # ===========================================

        # ============ 4 align the ground plane ============
        if cfg.align_with_ground:
            for i in range(T):
                #     t_align_z = np.mean(sbj_verts[i], axis=0)
                # else:
                z_min = min(np.min(sbj_verts[i, :, 2]), np.min(obj_verts[i, :, 2]))
                t_align_z = np.array([0.0, 0.0, -z_min], dtype=np.float32)

                preprocess_transforms[i]["t"] += t_align_z

                sbj_verts[i] += t_align_z
                obj_verts[i] += t_align_z
                sbj_joints[i] += t_align_z
                sbj_smpl["transl"][i] += t_align_z

                if cfg.behave.use_raw_pcs:
                    raw_pc_vertices[i] += t_align_z
        # t_align_z = 0
        # ==================================================

        # ============ 5 calculate rotation for object ============
        # load canonicalized object mesh
        obj_mesh = trimesh_load(can_objects_path / f"{seq_object}.ply")
        obj_vtemp = np.array(obj_mesh.vertices)
        obj_rotations, obj_centers = [], []
        for t_stamp in range(0, T):
            # find transform from vertices in the canonical pose to vertices in t_stamp frame
            R_t_stamp, t_t_stamp = estimate_transform(obj_vtemp, obj_verts[t_stamp])
            obj_rotations.append(R_t_stamp)
            obj_centers.append(t_t_stamp)
        # ===========================================

        # ============ 6 preprocess each time stamp in parallel
        # name mapping to split sequences for the same subject from different days
        if cfg.behave.split == "test":
            seq_subject = f"{seq_subject}_test"
        elif cfg.behave.split == "train":
            seq_subject = f"{seq_subject}_train"

        input_data = [{
            "sample": DatasetSample(
                subject=seq_subject,
                action=seq_action,
                object=seq_object,
                t_stamp=t,
                sbj_mesh=deepcopy(sbj_mesh),
                obj_mesh=deepcopy(obj_mesh),
                sbj_pc=sbj_verts[t],
                sbj_joints=sbj_joints[t],
                sbj_smpl={
                    "betas": sbj_smpl["betas"][t],
                    "transl": sbj_smpl["transl"][t],
                    "global_orient": sbj_smpl["global_orient"][t],
                    "body_pose": sbj_smpl["body_pose"][t],
                    "left_hand_pose": sbj_smpl["left_hand_pose"][t],
                    "right_hand_pose": sbj_smpl["right_hand_pose"][t]
                },
                obj_verts=obj_verts[t],
                obj_rotation=obj_rotations[t],
                obj_center=obj_centers[t],
                preprocess_transforms=preprocess_transforms[t]
            ),
            "num_points_pc_subject": cfg.num_points_pc_subject,
            "normalize": cfg.normalize
        } for t in range(T)]

        if cfg.behave.use_raw_pcs:
            raw_pc_vertices = [v for i, v in enumerate(raw_pc_vertices) if contact_mask[i]]
            for t in range(T):
                input_data[t]["raw_pc"] = raw_pc_vertices[t]

        # Same actions for each t_stamp
        preprocess_results = parallel_map(
            input_data,
            preprocess_worker_rawpc if cfg.behave.use_raw_pcs else preprocess_worker,
            use_kwargs=True, n_jobs=10, tqdm_kwargs={"leave": False}
        )
        # ===========================================

        # ============ 7 Save subject-specific data
        contact_masks[f"{seq_subject}_{seq_object}_{seq_action}"] = contact_mask
        seq_group_name = f"{seq_object}_{seq_action}"
        if seq_group_name in h5py_file[seq_subject]:
            del h5py_file[seq_subject][seq_group_name]
        seq_group = h5py_file[seq_subject].create_group(seq_group_name)
        add_sequence_datasets_to_hdf5(seq_group, preprocess_results[0], T)
        add_meatada_to_hdf5(seq_group, seq_subject, seq_object, seq_action, T, sequence_info["gender"])
        for sample in preprocess_results:
            sample.dump_hdf5(seq_group)
        # ===========================================

    # ============ 8 Save global info
    if cfg.behave.combine_1fps_and_30fps:
        suffix = f"{cfg.behave.split}" + "_" + '10fps' if cfg.behave.downsample else '30fps'

        with (target_folder / f"contact_masks_{suffix}.pkl").open("rb") as fp:
            contact_mask_30fps = pkl.load(fp)
        contact_masks.update(contact_mask_30fps)
    else:
        suffix = f"{cfg.behave.split}" + "_" + '1fps'
    with (target_folder / f"contact_masks_{suffix}.pkl").open("wb") as fp:
        pkl.dump(contact_masks, fp)

    OmegaConf.save(config=cfg, f=str(target_folder / f"preprocess_config_{suffix}_1fps.yaml"))
    # ===========================================



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess BEHAVE data with 30 fps annotations')

    parser.add_argument('--config', "-c", type=str, nargs="*", help='Path to YAML config(-s) file.')
    parser.add_argument("overrides", type=str, nargs="*", help="Overrides for the config.")
    arguments = parser.parse_args()

    config = init_preprocessing(arguments)

    # canonicalize objects using pre-computed transforms
    generate_behave_canonicalized_objects(
        Path(config.behave.orig_objects_path),
        Path(config.behave.can_objects_path)
    )
    # preprocess data
    preprocess(config)

    if config.behave.generate_obj_keypoints:
        generate_object_meshes(
            config.behave.objects,
            Path(config.behave.can_objects_path),
            Path(config.behave.target)
        )

        generate_obj_keypoints_from_barycentric(
            config.behave.objects,
            Path(config.env.assets_folder) / "object_keypoints" / "grab.pkl",
            Path(config.behave.target),
        )
