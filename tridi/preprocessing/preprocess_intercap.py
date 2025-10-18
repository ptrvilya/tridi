"""
Code to preprocess annotations for the InterCap dataset.
"""
import argparse
import json
import pickle as pkl
import shutil
from copy import deepcopy
from itertools import compress
from multiprocessing import set_start_method
from pathlib import Path

import h5py
import numpy as np
import smplx
import tqdm
import trimesh
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from .common import tensor_to_cpu, estimate_transform, DatasetSample, preprocess_worker, \
    contacts_worker, get_sequences_list, generate_obj_keypoints, trimesh_load, \
    add_sequence_datasets_to_hdf5, add_meatada_to_hdf5, init_preprocessing
from ..utils.parallel_map import parallel_map


def preprocess(cfg):
    set_start_method('spawn')

    # convert to Path
    root_folder = Path(cfg.intercap.root)
    target_folder = Path(cfg.intercap.target)
    objects_path = Path(cfg.intercap.objects_path)
    
    # list dataset sequences
    sequences = get_sequences_list(
        "intercap", root_folder, objects=cfg.intercap.objects, subjects=cfg.intercap.subjects
    )

    # init hdf5 file
    target_folder.mkdir(exist_ok=True, parents=True)
    if cfg.intercap.split in ["train", "test"]:
        hdf5_name = f"dataset_{cfg.intercap.split}"
    else:
        hdf5_name = "dataset"
    if cfg.intercap.downsample:
        hdf5_name += "_1fps"
    else:
        hdf5_name += "_10fps"

    if (target_folder / f"{hdf5_name}.hdf5").is_file():
        mode = "a"
    else:
        mode = "w"
    h5py_file = h5py.File(str(target_folder / f"{hdf5_name}.hdf5"), mode)

    # get subjects list and create groups
    subjects = list(set([s.parents[2].stem for s in sequences]))
    for sbj in subjects:
        if not sbj in h5py_file:
            h5py_file.create_group(sbj)

    # load gender mapping for smpl models
    with (root_folder / "gender.json").open("r") as fp:
        sbj2gender = json.load(fp)

    # preprocess each sequence
    contact_masks = {}
    for sequence in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        # parse sequence info
        seq_sbj = sequence.parents[2].stem
        seq_obj = sequence.parents[1].stem
        seq_obj_name = str(cfg.intercap.object_names[f"obj_{seq_obj}"])
        seq_action = sequence.parents[0].stem
        seq_gender = sbj2gender[f"sbj{seq_sbj}"]

        # ============ 1 extract vertices for subject
        if cfg.input_type != "smplh":
            raise ValueError(f"Unsupported input data type {cfg.input_type}")

        # load converted smplh model
        with sequence.open("rb") as fp:
            smplh_seq_data = pkl.load(fp)
        assert seq_gender == smplh_seq_data["gender"]
        # Mask marking frames kept during downsampling from 30 to 10 fps
        frame_mask = smplh_seq_data["frame_mask"]
        T_30fps = len(frame_mask)
        T_10fps = frame_mask.sum()
        # downsample further to 1fps
        if cfg.intercap.downsample:
            # create 10 fps mask
            mask_downsampled = np.zeros_like(frame_mask)
            mask_downsampled[::30] = True
            # obtain mask for smplh parameters (relative to 10fps)
            smplh_mask = np.zeros(T_10fps, dtype=bool)
            smplh_mask[mask_downsampled[np.argwhere(frame_mask)].flatten()] = True
            # obtain mask from 30 fps to 1 fps
            frame_mask = np.logical_and(frame_mask, mask_downsampled)
            # filter smplh params
            smplh_seq_data["body"] = {k: v[smplh_mask] for k, v in smplh_seq_data["body"].items()}
        T = frame_mask.sum()

        if T < 1:
            print("\nEmpty sequence", sequence)
            continue

        # generate new meshes
        sbj_model = smplx.build_layer(
            model_path=str(cfg.env.smpl_folder), model_type="smplh", gender=seq_gender,
            num_betas=10, batch_size=T, use_pca=False
        )
        body_model_params = {
            "betas": smplh_seq_data["body"]["betas"],
            "transl": smplh_seq_data["body"]["transl"],
            "global_orient": smplh_seq_data["body"]["global_orient"].reshape(T, -1, 9),
            "body_pose": smplh_seq_data["body"]["body_pose"].reshape(T, -1, 9),
            "left_hand_pose": smplh_seq_data["body"]["left_hand_pose"].reshape(T, -1, 9),
            "right_hand_pose": smplh_seq_data["body"]["right_hand_pose"].reshape(T, -1, 9),
        }

        # get smpl(-h) vertices
        sbj_output = sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
        sbj_verts = tensor_to_cpu(sbj_output.vertices)
        sbj_joints = tensor_to_cpu(sbj_output.joints)
        sbj_transl = body_model_params["transl"].numpy()
        sbj_orient = body_model_params["global_orient"].numpy().reshape(T, 3, 3)
        sbj_faces = sbj_model.faces

        # align sbj meshes
        preprocess_transforms =[]
        for i in range(T):
            R_grab = Rotation.from_euler('x', [-90], degrees=True)
            if cfg.align_with_joints:
                raise NotImplementedError("Aligning with joints is not supported.")
            else:
                # only align using center
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
        # ===========================================

        # ============ 2 extract vertices for object
        # load canonical template
        obj_mesh_template = trimesh_load(objects_path / f"obj_{seq_obj}.ply")
        obj_mesh_vtemp = np.array(obj_mesh_template.vertices)

        # filter object meshes
        original_mesh_folder = root_folder / "Res" / f"{seq_sbj}" / f"{seq_obj}" / f"{seq_action}" / "Mesh"
        obj_mesh_paths = sorted(list(original_mesh_folder.glob("*_second_obj.ply")))
        assert len(obj_mesh_paths) == T_30fps, f"Sequence {sequence} length mismatch {len(obj_mesh_paths)} vs {T_30fps}"
        obj_mesh_paths = list(compress(obj_mesh_paths, frame_mask))

        # load meshes (for better alignment compared to posing with params)
        # and transform using sbj transformation
        # transform using sbj transformation
        obj_verts = np.zeros((T, obj_mesh_vtemp.shape[0], 3), dtype=np.float32)
        for t_stamp in range(T):
            obj_mesh_t = trimesh_load(obj_mesh_paths[t_stamp])
            obj_mest_t_v = np.array(obj_mesh_t.vertices)

            R = Rotation.from_matrix(preprocess_transforms[t_stamp]["R"])
            t = preprocess_transforms[t_stamp]["t"]
            rot_center = preprocess_transforms[t_stamp]["rot_center"]
            obj_verts[t_stamp] = R.apply(obj_mest_t_v - rot_center) + rot_center + t
        # ===========================================

        # ============ 3 filter based on contacts ====
        input_data = [{
            "obj_mesh": deepcopy(obj_mesh_template),
            "obj_verts": obj_verts[t].astype(np.float32),
            "sbj_verts": sbj_verts[t].astype(np.float32),
            "sbj_faces": sbj_faces.astype(np.int32),
            "contact_threshold": cfg.intercap.contact_threshold
        } for t in range(T)]

        # Calculate contacts
        contact_mask = parallel_map(
            input_data, contacts_worker, use_kwargs=True, n_jobs=24, tqdm_kwargs={"leave": False}
        )
        contact_mask = np.array(contact_mask)
        T = contact_mask.sum()

        if T < 1:
            print("\nEmpty sequence after contacts filtering", sequence)
            continue

        sbj_verts = sbj_verts[contact_mask]
        obj_verts = obj_verts[contact_mask]
        sbj_joints = sbj_joints[contact_mask]
        sbj_smpl = {k: v[contact_mask] for k, v in sbj_smpl.items()}
        preprocess_transforms = list(compress(preprocess_transforms, contact_mask))
        # ============================================

        # ============ 4 align the ground plane ============
        if cfg.align_with_ground:
            raise NotImplementedError("Aligning with ground is not supported.")
        # ==================================================

        # ============ 5 calculate rotation for object ============
        # estimate transforms w.r.t. canonical template
        obj_rotations, obj_centers = [], []
        for t_stamp in range(0, T):
            # find transform from vertices in the canonical pose to vertices in t_stamp frame
            R_t_stamp, t_t_stamp = estimate_transform(obj_mesh_vtemp, obj_verts[t_stamp])
            obj_rotations.append(R_t_stamp)
            obj_centers.append(t_t_stamp)
        # ===========================================

        # ============ 6 preprocess each time stamp in parallel
        # create sbj mesh to use as a template
        sbj_mesh = trimesh.Trimesh(np.copy(sbj_verts[0]), sbj_faces, process=False)
        sbj_pointcloud = np.copy(sbj_verts)
        preprocess_results = []
        for t in tqdm.tqdm(range(T), leave=False, total=T, ncols=80):
            sample = DatasetSample(
                subject=seq_sbj,
                action=seq_action,
                object=seq_obj,
                t_stamp=t,
                sbj_mesh=deepcopy(sbj_mesh),
                obj_mesh=deepcopy(obj_mesh_template),
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

        # ============ 7 Save subject-specific data
        contact_masks[f"{seq_sbj}_{seq_obj_name}_{seq_action}"] = contact_mask
        seq_group_name = f"{seq_obj_name}_{seq_action}"
        if seq_group_name in h5py_file[seq_sbj]:
            del h5py_file[seq_sbj][seq_group_name]
        seq_group = h5py_file[seq_sbj].create_group(seq_group_name)
        add_sequence_datasets_to_hdf5(seq_group, preprocess_results[0], T)
        add_meatada_to_hdf5(seq_group, seq_sbj, seq_obj_name, seq_action, T, seq_gender)
        for sample in preprocess_results:
            sample.dump_hdf5(seq_group)
        # ===========================================

    # ============ 8 Save global info
    suffix = f"{cfg.intercap.split}" + "_" + '1fps' if cfg.intercap.downsample else '10fps'
    with (target_folder / f"contact_masks_{suffix}.pkl").open("wb") as fp:
        pkl.dump(contact_masks, fp)

    OmegaConf.save(config=cfg, f=str(target_folder / f"preprocess_config_{suffix}.yaml"))
    # ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess InterCap data')

    parser.add_argument('--config', "-c", type=str, nargs="*", help='Path to YAML config(-s) file.')
    parser.add_argument("overrides", type=str, nargs="*", help="Overrides for the config.")
    arguments = parser.parse_args()

    config = init_preprocessing(arguments)

    if not config.intercap.use_canonicalized_obj_meshes:
        raise NotImplementedError("Only canonicalized meshes are supported.")

    # define split
    if len(config.intercap.subjects) == 0:
        if config.intercap.split == "train":
            config.intercap.subjects = ["01", "02", "03", "04", "05", "06", "07", "08"]
        else:
            config.intercap.subjects = ["09", "10"]

    # preprocess data
    preprocess(config)
    # optionally generate object keypoints
    if config.intercap.generate_obj_keypoints:
        generate_obj_keypoints(
            config.intercap.objects, Path(config.intercap.objects_path),
            config.obj_keypoints_npoints, Path(config.intercap.target)
        )

        # rename objects
        objects = [f"{obj.stem}" for obj in list((Path(config.intercap.target) / "object_meshes").glob("*.ply"))]
        for obj in objects:
            name = str(config.intercap.object_names[f"{obj}"])
            shutil.move(Path(config.intercap.target) / "object_meshes" / f"{obj}.ply",
                        Path(config.intercap.target) / "object_meshes" / f"{name}.ply")
            shutil.move(Path(config.intercap.target) / "object_keypoints" / f"{obj}.npz",
                        Path(config.intercap.target) / "object_keypoints" / f"{name}.npz")