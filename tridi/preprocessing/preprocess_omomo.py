"""
Code to preprocess annotations for the OMOMO dataset.
"""
import argparse
import pickle as pkl
from copy import deepcopy
from itertools import compress
from multiprocessing import set_start_method
from pathlib import Path

import h5py
import joblib
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
    root_folder = Path(cfg.omomo.root)
    target_folder = Path(cfg.omomo.target)
    objects_path = Path(cfg.omomo.objects_path)
    
    # list dataset sequences
    sequences = get_sequences_list(
        "omomo", root_folder / "smplh" / cfg.omomo.split, objects=cfg.omomo.objects, subjects=cfg.omomo.subjects
    )

    # init hdf5 file
    target_folder.mkdir(exist_ok=True, parents=True)
    if cfg.omomo.split in ["train", "test"]:
        hdf5_name = f"dataset_{cfg.omomo.split}"
    else:
        raise ValueError(f"train or test must be provided as split, got {cfg.omomo.split}")
    if cfg.omomo.downsample:
        hdf5_name += "_1fps"
    else:
        hdf5_name += "_10fps"

    if (target_folder / f"{hdf5_name}.hdf5").is_file():
        mode = "a"
    else:
        mode = "w"
    h5py_file = h5py.File(str(target_folder / f"{hdf5_name}.hdf5"), mode)

    # Transform paths to pickle files into sequences' names
    sequences = [s.parent.name for s in sequences]

    # get subjects list and create groups
    subjects = list(set([s.split("_")[0] for s in sequences]))
    for sbj in subjects:
        if not sbj in h5py_file:
            h5py_file.create_group(sbj)

    # load annotations file
    sequences_data = joblib.load(
        root_folder / "data" / f"{cfg.omomo.split}_diffusion_manip_seq_joints24.p"
    )

    # load orig2can transforms for objects
    with open(root_folder / "tools" / "orig2can.pkl", "rb") as fp:
        obj_orig2can = pkl.load(fp)

    # preprocess each sequence
    contact_masks = {}
    print()
    for sequence_id in tqdm.tqdm(sequences_data.keys(), total=len(sequences_data.keys()), ncols=80):
        sequence_data = sequences_data[sequence_id]

        # check if we process this sequence
        seq_name = sequence_data["seq_name"]
        if not(seq_name in sequences):
            continue

        # parse sequence info
        seq_name_split = seq_name.split("_")
        seq_sbj = seq_name_split[0]
        seq_object = seq_name_split[1]
        seq_action = seq_name_split[2]
        seq_gender = sequence_data["gender"].item()

        # ============ 1 extract vertices for subject
        if cfg.input_type != "smplh":
            raise ValueError(f"Unsupported input data type {cfg.input_type}")

        # load converted smplh model
        smplh_seq_data_path = root_folder / "smplh" / cfg.omomo.split / seq_name / "sequence_data.pkl"
        with smplh_seq_data_path.open("rb") as fp:
            smplh_seq_data = pkl.load(fp)
        assert seq_gender == smplh_seq_data["gender"], f"{seq_gender} {smplh_seq_data['gender']}"
        # Mask marking frames kept during downsampling from 120 to 10 fps
        frame_mask = smplh_seq_data["frame_mask"]
        T_120fps = len(frame_mask)
        T_10fps = frame_mask.sum()
        # downsample further to 1fps
        if cfg.omomo.downsample:
            # create 10 fps mask
            mask_downsampled = np.zeros_like(frame_mask)
            mask_downsampled[::120] = True
            # obtain mask for smplh parameters (relative to 10fps)
            smplh_mask = np.zeros(T_10fps, dtype=bool)
            smplh_mask[mask_downsampled[np.argwhere(frame_mask)].flatten()] = True
            # obtain mask from 120 fps to 1 fps
            frame_mask = np.logical_and(frame_mask, mask_downsampled)
            # filter smplh params
            smplh_seq_data["body"] = {k: v[smplh_mask] for k, v in smplh_seq_data["body"].items()}
        T = frame_mask.sum()

        if T < 1:
            print("\nEmpty sequence", seq_name)
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
            R_grab = Rotation.from_euler('z', [45], degrees=True)
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
        # load transforms
        obj_rotations = sequence_data["obj_rot"]
        obj_centers = sequence_data["obj_trans"]

        # select orig2can transforms
        R_can = obj_orig2can[seq_object]["R"]
        t_can = obj_orig2can[seq_object]["t"]
        s_can = obj_orig2can[seq_object]["scale"]

        # load canonical mesh
        obj_mesh_template = trimesh_load(objects_path / f"{seq_object}.ply")
        obj_mesh_vtemp = np.array(obj_mesh_template.vertices)

        # filter transforms
        assert len(obj_rotations) == T_120fps, f"Sequence {seq_name} length mismatch {len(obj_rotations)} vs {T_120fps}"
        obj_rotations = obj_rotations[frame_mask].reshape(T, 3, 3)
        obj_centers = obj_centers[frame_mask].reshape(T, 1, 3)

        # obj_mesh_template = trimesh_load(root_folder / "data/captured_objects" / f"{seq_object}_cleaned_simplified.obj")
        # obj_mesh_vtemp = np.array(obj_mesh_template.vertices)
        # obj_rotations = s_can * obj_rotations

        # compute final transform
        # R = R_obj * R_can^(-1)
        # t = t_obj - R_obj * R_can^(-1) * (t_can * s_can)
        R_can_T = R_can.T.reshape(1, 3, 3).repeat(T, axis=0)
        obj_rotations = np.matmul(obj_rotations, R_can_T)
        _t_can_scaled = (t_can * s_can).reshape(1, 3).repeat(T, axis=0)
        obj_centers = obj_centers - np.matmul(obj_rotations, _t_can_scaled.reshape(T, 3, 1)).reshape(T, 1, 3)

        # transform canonical mesh to obtain object vertices
        obj_verts = np.copy(obj_mesh_vtemp).reshape(1, -1, 3).repeat(T, axis=0)
        obj_verts = \
            np.matmul(obj_rotations, obj_verts.transpose(0, 2, 1)).transpose(0, 2, 1) + \
            obj_centers.reshape(T, 1, 3)

        # transform again with subject's preprocessing
        for t_stamp in range(T):
            R = Rotation.from_matrix(preprocess_transforms[t_stamp]["R"])
            t = preprocess_transforms[t_stamp]["t"]
            rot_center = preprocess_transforms[t_stamp]["rot_center"]
            obj_verts[t_stamp] = R.apply(obj_verts[t_stamp] - rot_center) + rot_center + t
        # ===========================================

        # ============ 3 filter based on contacts ====
        input_data = [{
            "obj_mesh": deepcopy(obj_mesh_template),
            "obj_verts": obj_verts[t].astype(np.float32),
            "sbj_verts": sbj_verts[t].astype(np.float32),
            "sbj_faces": sbj_faces.astype(np.int32),
            "contact_threshold": cfg.omomo.contact_threshold
        } for t in range(T)]

        # Calculate contacts
        contact_mask = parallel_map(
            input_data, contacts_worker, use_kwargs=True, n_jobs=24, tqdm_kwargs={"leave": False}
        )
        contact_mask = np.array(contact_mask)
        T = contact_mask.sum()

        if T < 1:
            print("\nEmpty sequence after contacts filtering", seq_name)
            continue

        sbj_verts = sbj_verts[contact_mask]
        obj_verts = obj_verts[contact_mask]
        sbj_joints = sbj_joints[contact_mask]
        sbj_smpl = {k: v[contact_mask] for k, v in sbj_smpl.items()}
        obj_rotations = obj_rotations[contact_mask]
        obj_centers = obj_centers[contact_mask]
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
                object=seq_object,
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
        contact_masks[seq_name] = contact_mask
        if f"{seq_object}_{seq_action}" in h5py_file[seq_sbj]:
            del h5py_file[seq_sbj][f"{seq_object}_{seq_action}"]
        seq_group = h5py_file[seq_sbj].create_group(f"{seq_object}_{seq_action}")
        add_sequence_datasets_to_hdf5(seq_group, preprocess_results[0], T)
        add_meatada_to_hdf5(seq_group, seq_sbj, seq_object, seq_action, T, seq_gender)
        for sample in preprocess_results:
            sample.dump_hdf5(seq_group)
        # ===========================================

    # ============ 8 Save global info
    suffix = f"{cfg.omomo.split}" + "_" + '1fps' if cfg.omomo.downsample else '10fps'
    contact_mask_path = target_folder / f"contact_masks_{suffix}.pkl"
    if contact_mask_path.is_file():
        with contact_mask_path.open("rb") as fp:
            contact_masks_old = pkl.load(fp)
            contact_masks.update(contact_masks_old)

    with contact_mask_path.open("wb") as fp:
        pkl.dump(contact_masks, fp)

    OmegaConf.save(config=cfg, f=str(target_folder / f"preprocess_config_{suffix}.yaml"))
    # ===========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess OMOMO data')

    parser.add_argument('--config', "-c", type=str, nargs="*", help='Path to YAML config(-s) file.')
    parser.add_argument("overrides", type=str, nargs="*", help="Overrides for the config.")
    arguments = parser.parse_args()

    config = init_preprocessing(arguments)

    if not config.omomo.use_canonicalized_obj_meshes:
        raise NotImplementedError("Only canonicalized meshes are supported.")

    # define split
    if len(config.omomo.subjects) == 0:
        if config.omomo.split == "train":
            config.omomo.subjects = [f"sub{s}" for s in range(1, 16)]
        else:
            config.omomo.subjects = ["sub16", "sub17"]

    # preprocess data
    preprocess(config)
    # optionally generate object keypoints
    if config.omomo.generate_obj_keypoints:
        generate_obj_keypoints(
            config.omomo.objects, Path(config.omomo.objects_path),
            config.obj_keypoints_npoints, Path(config.omomo.target)
        )