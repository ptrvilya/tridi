"""
Code to preprocess annotations for the GRAB dataset.
"""
import argparse
import pickle as pkl
import warnings
from copy import deepcopy
from multiprocessing import set_start_method
from pathlib import Path

import h5py
import numpy as np
import smplx
import tqdm
import trimesh
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from .common import (tensor_to_cpu, estimate_transform, DatasetSample, preprocess_worker, \
    parse_npz, get_sequences_list, generate_object_meshes, prepare_params, \
    generate_obj_keypoints_from_barycentric, init_preprocessing,
    trimesh_load, add_sequence_datasets_to_hdf5, add_meatada_to_hdf5, params_to_torch)
from .grab_object_model import ObjectModel


def preprocess(cfg):
    set_start_method('spawn')

    # convert to Path
    root_folder = Path(cfg.grab.root)
    target_folder = Path(cfg.grab.target)

    # list dataset sequences
    sequences = get_sequences_list(
        "grab", root_folder, objects=cfg.grab.objects, subjects=cfg.grab.subjects
    )

    # init hdf5 file
    target_folder.mkdir(exist_ok=True, parents=True)
    if cfg.grab.split in ["train", "test"]:
        hdf5_name = f"dataset_{cfg.grab.split}"
    else:
        hdf5_name = "dataset"
    if cfg.grab.downsample != "None":
        hdf5_name += f"_{cfg.grab.downsample}"

    if (target_folder / f"{hdf5_name}.hdf5").is_file():
        mode = "a"
    else:
        mode = "w"
    h5py_file = h5py.File(str(target_folder / f"{hdf5_name}.hdf5"), mode)
    for sbj in cfg.grab.subjects:
        if not sbj in h5py_file:
            h5py_file.create_group(sbj)

    # preprocess each sequence
    for sequence in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        # Parse original grab annotations
        grab_seq_data = parse_npz(sequence)
        seq_subject = str(grab_seq_data["sbj_id"])
        seq_action = "_".join(sequence.stem.split("_")[1:])
        seq_object = str(grab_seq_data["obj_name"])

        # ============ 1 extract vertices for subject
        preprocess_transforms = []
        if cfg.input_type in ["smplh", "smpl"]:
            # path to custom converted smplh meshes
            seq_name = f"{seq_object}_{seq_action}"
            smplh_sequence_folder = root_folder / "grab_smplh" / seq_subject / seq_name

            # load sequence data for smplh
            with (smplh_sequence_folder / "sequence_data.pkl").open("rb") as fp:
                smplh_seq_data = pkl.load(fp)

            # frame mask for smplh meshes
            frame_mask = smplh_seq_data["frame_mask"]
            T_init = frame_mask.sum()

            # initial mask is at 30 fps
            # NOTE: smplh annotations are downsampled from 120 fps to 30 fps
            smplh_mask = np.ones(T_init, dtype=bool)
            if cfg.grab.downsample != "None":
                downsample_mask = np.zeros_like(frame_mask)
                if cfg.grab.downsample == "10fps":
                    downsample_mask[::12] = True
                elif cfg.grab.downsample == "1fps":
                    downsample_mask[::120] = True
                else:
                    downsample_mask[::4] = True

                # downsample masks
                smplh_mask = downsample_mask[np.argwhere(frame_mask)].flatten()
                frame_mask = np.logical_and(frame_mask, downsample_mask)
            T = frame_mask.sum()

            # if no frame is selected continue to the next sequence
            if T < 1:
                continue

            # generate new meshes
            sbj_model = smplx.build_layer(
                model_path=str(cfg.env.smpl_folder), model_type="smplh", gender=grab_seq_data["gender"],
                num_betas=10, batch_size=T, use_pca=False
            )
            body_model_params = {
                "betas": smplh_seq_data["body"]["betas"][smplh_mask],
                "transl": smplh_seq_data["body"]["transl"][smplh_mask],
                "global_orient": smplh_seq_data["body"]["global_orient"][smplh_mask].reshape(T, -1, 9),
                "body_pose": smplh_seq_data["body"]["body_pose"][smplh_mask].reshape(T, -1, 9),
                "left_hand_pose": smplh_seq_data["body"]["left_hand_pose"][smplh_mask].reshape(T, -1, 9),
                "right_hand_pose": smplh_seq_data["body"]["right_hand_pose"][smplh_mask].reshape(T, -1, 9),
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
            for i in range(T):
                if cfg.align_with_joints:
                    old_transl = sbj_transl[i]
                    pelvis = sbj_joints[i][0] - old_transl
                    rot_center = pelvis + old_transl

                    # remove effect of global transl and center in pelvis
                    sbj_joints[i] = sbj_joints[i] - rot_center
                    sbj_verts[i] = sbj_verts[i] - rot_center

                    # rotation to align torso with x axis direction
                    shoulderblades = sbj_joints[i][2] - sbj_joints[i][1]
                    z = np.array([0, 0, 1])
                    dir_shoulderblades = np.cross(z, shoulderblades)
                    dir_shoulderblades[2] = 0.0  # project to xy
                    dir_shoulderblades = dir_shoulderblades / np.linalg.norm(dir_shoulderblades)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        R_align, _ = Rotation.align_vectors(np.array([[1, 0, 0]]), dir_shoulderblades[None])

                    # rotate around pelvis
                    sbj_verts[i] = R_align.apply(sbj_verts[i])
                    sbj_joints[i] = R_align.apply(sbj_joints[i])

                    # reapply pelvis and global translation
                    sbj_joints[i] = sbj_joints[i] + rot_center
                    sbj_verts[i] = sbj_verts[i] + rot_center

                    t_reset = -1 * np.copy(sbj_joints[i][0])  # center using root joint
                    sbj_joints[i] += t_reset
                    sbj_verts[i] += t_reset

                    # new global smpl params
                    sbj_orient[i] = (R_align * Rotation.from_matrix(sbj_orient[i])).as_matrix()
                    sbj_transl[i] += t_reset
                else:
                    # only align using center
                    R_align = Rotation.from_matrix(np.eye(3, dtype=np.float32))
                    old_transl = sbj_transl[i]
                    pelvis = sbj_joints[i][0] - old_transl
                    rot_center = pelvis + old_transl

                    # t_reset = -1 * np.mean(sbj_verts[i], axis=0)
                    t_reset = -1 * sbj_joints[i][0]
                    sbj_joints[i] += t_reset
                    sbj_verts[i] += t_reset

                    # new global smpl params, Rotation is unchanged
                    sbj_transl[i] += t_reset

                preprocess_transforms.append({
                    "R": R_align.as_matrix(),
                    "t": np.copy(t_reset),
                    "rot_center": np.copy(rot_center)
                })

            # save smpl parameters
            sbj_smpl = {
                "betas": body_model_params["betas"],
                "transl": sbj_transl,
                "global_orient": sbj_orient.reshape(T, 1, 9),
                "body_pose": body_model_params["body_pose"].reshape(T, -1, 9).numpy(),
                "left_hand_pose": body_model_params["left_hand_pose"].reshape(T, -1, 9).numpy(),
                "right_hand_pose": body_model_params["right_hand_pose"].reshape(T, -1, 9).numpy()
            }

            # create template mesh
            sbj_faces = sbj_model.faces
            sbj_mesh = trimesh.Trimesh(vertices=sbj_verts[0], faces=sbj_faces)
        else:
            raise ValueError(f"Unsupported input data type {cfg.input_type}")
        # ===========================================

        # ============ 2 extract vertices for object
        # parse obj parameters from annotations
        obj_params = prepare_params(grab_seq_data["object"]["params"], frame_mask)

        # pose all meshes in the sequence using ObjectModel
        obj_full_mesh = str(root_folder / grab_seq_data["object"]["object_mesh"])
        obj_full_mesh = trimesh_load(obj_full_mesh)
        obj_full_vtemp = np.array(obj_full_mesh.vertices)
        obj_full_model = ObjectModel(v_template=obj_full_vtemp, batch_size=T)
        obj_full_parms = params_to_torch(obj_params)
        obj_full_verts = tensor_to_cpu(obj_full_model(**obj_full_parms).vertices)

        # transform using sbj transformation
        for index in range(T):
            obj_full_v = obj_full_verts[index]

            R = Rotation.from_matrix(preprocess_transforms[index]["R"])
            t = preprocess_transforms[index]["t"]
            rot_center = preprocess_transforms[index]["rot_center"]
            obj_full_verts[index] = R.apply(obj_full_v - rot_center) + rot_center + t

        # optionally load decimated mesh
        if cfg.grab.use_decimated_obj_meshes:
            obj_mesh = root_folder / grab_seq_data["object"]["object_mesh"]
            obj_mesh = obj_mesh.parents[1] / "decimated_meshes" / obj_mesh.name
            obj_mesh = trimesh_load(obj_mesh)
            obj_dec_vtemp = np.array(obj_mesh.vertices)
            obj_dec_verts = np.tile(np.copy(obj_dec_vtemp), (T, 1, 1))
        else:
            obj_mesh = obj_full_mesh
        # calculate obj rotations and optionally transform decimated mesh for each t_stamp
        obj_rotations, obj_centers = [], []
        for t_stamp in range(0, T):
            # find transform from vertices in the canonical pose to vertices in t_stamp frame
            R_t_stamp, t_t_stamp = estimate_transform(obj_full_vtemp, obj_full_verts[t_stamp])
            obj_rotations.append(R_t_stamp)
            obj_centers.append(t_t_stamp)

            # THIS STEP IS OPTIONAL, USED TO SAVE SPACE
            if cfg.grab.use_decimated_obj_meshes:
                # apply transforms to decimated mesh to obtain posed decimated mesh for all frames
                obj_dec_verts[t_stamp] = np.dot(obj_dec_verts[t_stamp], R_t_stamp.T) + t_t_stamp[None]
        if cfg.grab.use_decimated_obj_meshes:
            obj_verts = obj_dec_verts
        else:
            obj_verts = obj_full_verts
        # ===========================================

        # ============ 3 align the ground plane ============
        if cfg.align_with_ground and cfg.input_type in ["smplh", "smpl"]:
            for i in range(T):
                # if cfg.normalize:
                #     t_align_z = np.mean(sbj_verts[i], axis=0)
                # else:
                z_min = min(np.min(sbj_verts[i, :, 2]), np.min(obj_verts[i, :, 2]))
                t_align_z = np.array([0.0, 0.0, -z_min], dtype=np.float32)

                preprocess_transforms[i]["t"] += t_align_z

                sbj_verts[i] += t_align_z
                obj_verts[i] += t_align_z
                sbj_joints[i] += t_align_z
                sbj_smpl["transl"][i] += t_align_z
                obj_centers[i] += t_align_z
        # ==================================================

        # ============ 4 preprocess each time stamp in parallel
        sbj_pointcloud = np.copy(sbj_verts)
        preprocess_results = []
        for t in tqdm.tqdm(range(T), leave=False, total=T, ncols=80):
            sample = DatasetSample(
                subject=seq_subject,
                action=seq_action,
                object=seq_object,
                t_stamp=t,
                sbj_mesh=deepcopy(sbj_mesh),
                obj_mesh=deepcopy(obj_mesh),
                sbj_pc=sbj_pointcloud[t],
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
            )

            # Same actions for each t_stamp
            preprocess_results.append(preprocess_worker(sample, cfg.normalize))
        # ===========================================

        # ============ 5 Save subject-specific data
        seq_group_name = f"{seq_object}_{seq_action}"
        if seq_group_name in h5py_file[seq_subject]:
            del h5py_file[seq_subject][seq_group_name]
        seq_group = h5py_file[seq_subject].create_group(seq_group_name)
        add_sequence_datasets_to_hdf5(seq_group, preprocess_results[0], T)
        add_meatada_to_hdf5(seq_group, seq_subject, seq_object, seq_action, T, grab_seq_data["gender"])
        for sample in preprocess_results:
            sample.dump_hdf5(seq_group)
        # ===========================================

    # ============ 6 Save global info
    config_filename = f"preprocess_config_{cfg.grab.split}_{cfg.grab.downsample}.yaml"
    OmegaConf.save(config=cfg, f=str(target_folder / config_filename))
    # ===========================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data')

    parser.add_argument('--config', "-c", type=str, nargs="*", help='Path to YAML config(-s) file.')
    parser.add_argument("overrides", type=str, nargs="*", help="Overrides for the config.")
    arguments = parser.parse_args()

    config = init_preprocessing(arguments)

    if not config.grab.use_decimated_obj_meshes:
        config.grab.objects_path = str(Path(config.grab.root) / "tools/object_meshes/contact_meshes")

    preprocess(config)

    if config.grab.generate_obj_keypoints:
        generate_object_meshes(
            config.grab.objects,
            Path(config.grab.objects_path),
            Path(config.grab.target)
        )

        generate_obj_keypoints_from_barycentric(
            config.behave.objects,
            Path(config.env.assets_folder) / "object_keypoints" / "grab.pkl",
            Path(config.grab.target),
        )