import argparse
from pathlib import Path
from types import SimpleNamespace
import pickle as pkl
import json
from collections import defaultdict
import joblib

import tqdm
import smplx
import numpy as np
import trimesh
import open3d as o3d
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation

from smplx import build_layer

from config import omegaconf_from_dict
from transfer_model import run_fitting
from utils import read_deformation_transfer, np_mesh_to_o3d
from utils import batch_rodrigues


def filter_contact_frames(cfg, seq_data):
    if cfg.only_contact:
        frame_mask = (seq_data['contact']['object'] > 0).any(axis=1)
    else:
        frame_mask = (seq_data['contact']['object'] > -1).any(axis=1)
    return frame_mask


def parse_sequence(sequence):
    T = len(sequence["trans"])

    body_pose = torch.tensor(sequence["pose_body"]).reshape(T, 21, 3)
    sbj_params = {
        "betas": sequence["betas"].reshape(1, 16).repeat(T, 0),
        "transl": sequence["trans"],
        "global_orient": batch_rodrigues(torch.tensor(sequence["root_orient"])).reshape(T, 1, 9).numpy(),
        "body_pose": batch_rodrigues(body_pose.reshape(T*21, 3)).reshape(T, 21, 9).numpy(),
    }
    # sbj_params['body_pose'] = sbj_params['body_pose'].reshape(T, 21, 3)
    # obj_params = {
    #     k: res2_data[f"ob_{k}"] for k in ['pose', 'trans']
    # }

    seq_info = {
        "T": T,
        'seq_name': sequence['seq_name'],
    }

    return seq_info, sbj_params


def prepare_params(params, frame_mask, dtype=np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}


def params2torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


class MeshInMemory(Dataset):
    def __init__(self, faces, verts, all_samples_info):
        self.faces = faces
        self.verts = verts
        self.seq_info = all_samples_info
        self.num_items = len(self.verts)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):
        mesh_verts = self.verts[index]

        return {
            'vertices': np.asarray(mesh_verts, dtype=np.float32),
            'faces': np.copy(self.faces).astype(np.int32),
            'index': index,
            'seq_name': self.seq_info[index]['seq_name'],
            't_stamp': self.seq_info[index]['t'],
            'output_path': str(self.seq_info[index]['output_seq_path']),
        }


def create_smplx_meshes(seq_info, sbj_params, cfg):
    # frame_mask = filter_contact_frames(cfg, seq_data)

    T = seq_info["T"]
    if cfg.downsample:  # downsample from 120fps to 10fps
        frame_mask = np.zeros(T, dtype=bool)
        frame_mask[::12] = True
        frame_mask[-1] = True
    T = frame_mask.sum()

    # create SMPL-X mesh
    sbj_model = smplx.build_layer(
        model_path=str(cfg.model_path), model_type='smplx',
        gender=seq_info["gender"], num_betas=16, batch_size=T,
        use_pca=False, flat_hand_mean=True,
    )

    sbj_params = {k: v[frame_mask] for k, v in sbj_params.items()}

    sbj_parms = params2torch(sbj_params)
    # dict_keys(['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
    # 'left_hand_pose', 'right_hand_pose', 'fullpose', 'expression'])
    sbj_parms["left_hand_pose"] = None
    sbj_parms["right_hand_pose"] = None
    sbj_output = sbj_model(**sbj_parms)
    sbj_verts = to_cpu(sbj_output.vertices)

    return sbj_model.faces, sbj_verts, frame_mask


def run_smplx2smplh_conversion(dataset, seq2info, cfg, gender):
    exp_conf = {
        "deformation_transfer_path": "./transfer_data/smplx2smplh_deftrafo_setup.pkl",
        "mask_ids_fname": '',
        "summary_steps": 100,

        "edge_fitting": {"per_part": False},

        "optim": {"type": 'lbfgs', "maxiters": 150, "gtol": 1e-06},
        "batch_size": cfg.batch_size,

        "body_model": {
            "model_type": "smplh",
            # SMPL+H has no neutral model, so we have to manually select the gender
            "gender": gender,
            "ext": 'pkl',
            "folder": cfg.model_path,
            "use_compressed": False,
            "use_face_contour": True,
            "smplh": {"betas": {"num": 10}}
        }
    }

    device = torch.device("cuda:0")
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
    )
    exp_omegaconf = omegaconf_from_dict(exp_conf)

    model_path = exp_omegaconf.body_model.folder
    body_model = build_layer(model_path, **exp_omegaconf.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)
    mask_ids = None

    deformation_transfer_path = exp_omegaconf.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    res_smplh = dict()
    for seq_name in seq2info.keys():
        T = seq2info[seq_name]["T"]
        res_smplh[seq_name] = {
            "gender": gender,
            "frame_mask": seq2info[seq_name]["frame_mask"],
            "body": {
                'transl': torch.zeros((T, 3), dtype=torch.float),
                'global_orient': torch.zeros((T, 1, 3, 3), dtype=torch.float),
                'body_pose': torch.zeros((T, 21, 3, 3), dtype=torch.float),
                'betas': torch.zeros((T, 10), dtype=torch.float),
                'left_hand_pose': torch.zeros((T, 15, 3, 3), dtype=torch.float),
                'right_hand_pose': torch.zeros((T, 15, 3, 3), dtype=torch.float),
                'full_pose': torch.zeros((T, 52, 3, 3), dtype=torch.float),
            }
        }

    body_keys = ['transl', 'global_orient', 'body_pose', 'betas', 'left_hand_pose', 'right_hand_pose', 'full_pose']
    # body = {
    #     'transl': [], 'global_orient': [], 'body_pose': [], 'betas': [], 'left_hand_pose': [], 'right_hand_pose': [],
    #     'full_pose': []
    # }

    # sbj_model = smplx.build_layer(
    #     model_path=str(cfg.model_path), model_type="smplh", gender=seq_info['gender'],
    #     num_betas=10, batch_size=cfg.batch_size, num_pca_comps=12, use_pca=False, use_compressed=False
    # ).to(device)

    for batch_index, batch in enumerate(tqdm.tqdm(dataloader)):
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)
        var_dict = run_fitting(exp_omegaconf, batch, body_model, def_matrix, mask_ids)

        t_stamps = batch['t_stamp'].detach().cpu().numpy().tolist()
        vertices = var_dict.pop("vertices")
        faces = var_dict.pop("faces")
        var_dict.pop("v_shaped"), var_dict.pop("joints")

        # optionally save meshes
        if cfg.save_meshes:
            for i, t_stamp in enumerate(t_stamps):
                output_mesh_path = str(Path(batch["output_path"][i]) / f"{t_stamp:04d}.obj")
                mesh_smplh = np_mesh_to_o3d(vertices[i], faces)
                o3d.io.write_triangle_mesh(output_mesh_path, mesh_smplh)

                output_mesh_path = str(Path(batch["output_path"][i]) / f"smplx_{t_stamp:04d}.obj")
                v = batch['vertices'][i].detach().cpu().numpy()
                f = batch['faces'][i].detach().cpu().numpy()
                mesh_smplx = np_mesh_to_o3d(v, f)
                o3d.io.write_triangle_mesh(output_mesh_path, mesh_smplx)

        # save params per sequence
        var_dict = {k: v.detach().cpu() for k, v in var_dict.items()}
        for i, t_stamp in enumerate(t_stamps):
            seq_name = batch['seq_name'][i]
            for k in body_keys:
                res_smplh[seq_name]["body"][k][t_stamp] = var_dict[k][i]

        # # save parameters per-batch
        # for k in body.keys():
        #     body[k].append(var_dict[k].detach().cpu())

    # # concatenate per-batch
    # for k in body.keys():
    #     body[k] = torch.cat(body[k], dim=0)

    for seq_name in seq2info.keys():
        # convert pose matrices to rotvec
        body = res_smplh[seq_name]["body"]
        matrices_blhrh = [body["body_pose"], body["left_hand_pose"], body["right_hand_pose"]]
        matrices_blhrh = torch.cat(matrices_blhrh, dim=1).numpy()
        rotvec_blhrh = np.zeros(matrices_blhrh.shape[:3], dtype=np.float32)
        for t in range(matrices_blhrh.shape[0]):
            R = Rotation.from_matrix(matrices_blhrh[t])
            rotvec_blhrh[t] = R.as_rotvec()

        body["pose_blhrh_rotvec"] = rotvec_blhrh

        # save converted data
        sequence_data_path = seq2info[seq_name]["output_seq_path"] / f"sequence_data.pkl"
        with sequence_data_path.open('wb') as fp:
            pkl.dump(res_smplh[seq_name], fp)
        print(f"saved {sequence_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conversion of OMOMO annotations from SMPL-X to SMPL+H.')

    parser.add_argument('-i', "--omomo-path", type=Path)
    parser.add_argument('-s', "--smplx-path", type=Path)

    args = parser.parse_args()

    # Adjust parameters manually
    cfg = {
        # 'only_contact': True,
        'omomo_root': Path(args.omomo_path),

        "split": "train",  # "train"

        'batch_size': 2048,
        'num_workers': 6,

        'downsample': True,  # downsample from 120 fps to 10 fps
        'save_meshes': False,

        # body and hand model path
        'model_path': args.smplx_path,
        'transfer_data_path': "./transfer_data",
    }

    cfg = SimpleNamespace(**cfg)

    data = joblib.load(cfg.omomo_root / "data" / f"{cfg.split}_diffusion_manip_seq_joints24.p")

    all_seq_ids = list(data.keys())

    # split sequences per gender to optimize smpl model calls
    all_seqs_per_subject = defaultdict(list)
    subject2genderlist = defaultdict(list)
    for sequence_id in all_seq_ids:
        seq_name = data[sequence_id]["seq_name"]
        seq_subject = seq_name.split("_")[0]

        all_seqs_per_subject[seq_subject].append(sequence_id)
        subject2genderlist[seq_subject].append(data[sequence_id]["gender"].item())

    subject2gender = dict()
    for subject, genders in subject2genderlist.items():
        assert genders == len(genders) * [genders[0]]
        subject2gender[subject] = genders[0]

    genders = set(subject2gender.values())
    # process sequences per gender
    for gender in genders:
        for subject in all_seqs_per_subject.keys():
            if subject2gender[subject] != gender:
                continue

            seq2info = dict()
            all_vertices, all_samples_info = [], []
            faces = None

            print(f"Loading data for subject: {subject}")
            for sequence_id in tqdm.tqdm(
                    all_seqs_per_subject[subject], total=len(all_seqs_per_subject[subject]), ncols=80
            ):
                sbj_gender = subject2gender[subject]
                seq_info, sbj_params = parse_sequence(data[sequence_id])

                seq_info["gender"] = sbj_gender
                seq_info["output_seq_path"] = \
                    cfg.omomo_root / "smplh" / f"{cfg.split}" / f"{seq_info['seq_name']}"
                seq_info["output_seq_path"].mkdir(exist_ok=True, parents=True)
                faces, verts, frame_mask = create_smplx_meshes(seq_info, sbj_params, cfg)
                seq2info[seq_info['seq_name']] = {
                    "output_seq_path": seq_info["output_seq_path"],
                    "T": frame_mask.sum(),
                    'frame_mask': frame_mask
                }

                all_vertices.append(verts)
                samples_info = []
                for t in range(seq2info[seq_info['seq_name']]["T"]):
                    samples_info.append({
                        "t": t,
                        "gender": seq_info["gender"],
                        "seq_name": seq_info["seq_name"],
                        "output_seq_path": seq_info["output_seq_path"],
                    })
                all_samples_info.extend(samples_info)

                assert len(samples_info) == len(verts)

            print(f"Converting: {subject}")
            all_vertices = np.concatenate(all_vertices, axis=0)

            mesh_dataset = MeshInMemory(faces, all_vertices, all_samples_info)
            run_smplx2smplh_conversion(mesh_dataset, seq2info, cfg, sbj_gender)
