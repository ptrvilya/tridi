import argparse
from pathlib import Path
from types import SimpleNamespace
import pickle as pkl
import json

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


def filter_contact_frames(cfg, seq_data):
    if cfg.only_contact:
        frame_mask = (seq_data['contact']['object'] > 0).any(axis=1)
    else:
        frame_mask = (seq_data['contact']['object'] > -1).any(axis=1)
    return frame_mask


def parse_sequence(sequence):
    with open(sequence, "rb") as fp:
        res2_data = pkl.load(fp)

    sbj_params = {
        k: res2_data[k] for k in [
            'betas', 'transl', 'global_orient', 'body_pose',
            # 'jaw_pose', 'leye_pose', 'reye_pose',
            'left_hand_pose', 'right_hand_pose',
            'expression'
        ]
    }

    T = len(sbj_params['betas'])
    sbj_params['body_pose'] = sbj_params['body_pose'].reshape(T, 21, 3)
    # obj_params = {
    #     k: res2_data[f"ob_{k}"] for k in ['pose', 'trans']
    # }

    seq_info = {
        "T": T,
        "sbj_id": sequence.parents[2].stem,
        "action": sequence.parents[0].stem,
        "obj_id": sequence.parents[1].stem
    }

    return seq_info, sbj_params


def prepare_params(params, frame_mask, dtype = np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}


def params2torch(params, dtype = torch.float32):
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
            't_stamp': self.seq_info[index]['t'],
            'path': f"{self.seq_info[index]['sbj_id']}/"
                     f"{self.seq_info[index]['obj_id']}/{self.seq_info[index]['action']}/"
                     f"{index:04d}.obj",
            'output_path': str(self.seq_info[index]['output_seq_path']),
            'sbj_id': self.seq_info[index]['sbj_id'],
            'obj_id': self.seq_info[index]['obj_id'],
            'action': self.seq_info[index]['action']
        }


def create_smplx_meshes(seq_info, sbj_params, cfg):
    # frame_mask = filter_contact_frames(cfg, seq_data)

    T = seq_info["T"]
    if cfg.downsample:  # downsample from 30fps to 10fps
        frame_mask = np.zeros(T, dtype=bool)
        frame_mask[::3] = True
    T = frame_mask.sum()

    # create SMPL-X mesh
    sbj_model = smplx.create(
        model_path=str(cfg.model_path), model_type='smplx',
        gender=seq_info["gender"], is_rhand=False, num_pca_comps=12,
        flat_hand_mean=False, use_pca=True, batch_size=T
    )

    sbj_params = {k: v[frame_mask] for k, v in sbj_params.items()}

    sbj_parms = params2torch(sbj_params)
    # dict_keys(['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
    # 'left_hand_pose', 'right_hand_pose', 'fullpose', 'expression'])
    sbj_output = sbj_model(**sbj_parms)
    sbj_verts = to_cpu(sbj_output.vertices)

    return sbj_model.faces, sbj_verts, frame_mask


def run_smplx2smplh_conversion(dataset, seq2T, cfg, gender):
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
    for sbj_id in seq2T.keys():
        res_smplh[sbj_id] = dict()
        for obj_id in seq2T[sbj_id].keys():
            res_smplh[sbj_id][obj_id] = dict()
            for action in seq2T[sbj_id][obj_id].keys():
                T = seq2T[sbj_id][obj_id][action]["T"]
                res_smplh[sbj_id][obj_id][action] = {
                    "gender": gender,
                    "sbj_id": sbj_id,
                    "obj_id": obj_id,
                    "action": action,
                    "frame_mask": seq2T[sbj_id][obj_id][action]["frame_mask"],
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
            sbj_id = batch['sbj_id'][i]
            obj_id = batch['obj_id'][i]
            action = batch['action'][i]
            for k in body_keys:
                res_smplh[sbj_id][obj_id][action]["body"][k][t_stamp] = var_dict[k][i]

        # # save parameters per-batch
        # for k in body.keys():
        #     body[k].append(var_dict[k].detach().cpu())

    # # concatenate per-batch
    # for k in body.keys():
    #     body[k] = torch.cat(body[k], dim=0)

    for sbj_id in seq2T.keys():
        for obj_id in seq2T[sbj_id].keys():
            for action, T in seq2T[sbj_id][obj_id].items():
                # convert pose matrices to rotvec
                body = res_smplh[sbj_id][obj_id][action]["body"]
                matrices_blhrh = [body["body_pose"], body["left_hand_pose"], body["right_hand_pose"]]
                matrices_blhrh = torch.cat(matrices_blhrh, dim=1).numpy()
                rotvec_blhrh = np.zeros(matrices_blhrh.shape[:3], dtype=np.float32)
                for t in range(matrices_blhrh.shape[0]):
                    R = Rotation.from_matrix(matrices_blhrh[t])
                    rotvec_blhrh[t] = R.as_rotvec()

                body["pose_blhrh_rotvec"] = rotvec_blhrh

                # save converted data
                sequence_data_path = seq2T[sbj_id][obj_id][action]["output_seq_path"] / f"sequence_data.pkl"
                with sequence_data_path.open('wb') as fp:
                    pkl.dump(res_smplh[sbj_id][obj_id][action], fp)
                print(f"saved {sequence_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conversion of InterCap annotations from SMPL-X to SMPL+H.')

    parser.add_argument('-i', "--intercap-sequences", type=Path)
    parser.add_argument('-r', "--intercap-root", type=Path)
    parser.add_argument('-s', "--smplx-path", type=Path)

    args = parser.parse_args()

    # Adjust parameters manually
    cfg = {
        # 'only_contact': True,
        'intercap_sequences': Path(args.intercap_sequences),
        'intercap_root': Path(args.intercap_root),

        'batch_size': 2048,
        'num_workers': 6,

        'downsample': True,  # downsample from 30 fps to 10 fps
        'save_meshes': False,


        # body and hand model path
        'model_path': args.smplx_path,
        'transfer_data_path': "./transfer_data",
    }

    cfg = SimpleNamespace(**cfg)
    all_seqs = list(cfg.intercap_sequences.glob('*/*/Seg*/res_2.pkl'))

    with (cfg.intercap_root / "gender.json").open("r") as fp:
        sbj2gender = json.load(fp)
    gender2sbj = {
        g: [k for k, v in sbj2gender.items() if v == g] for g in ['male', 'female']
    }
    all_seqs_per_gender = {
        'male': [], 'female': []
    }
    for sequence in all_seqs:
        sbj_id = sequence.parents[2].stem
        all_seqs_per_gender[sbj2gender[f"sbj{sbj_id}"]].append(sequence)

    for gender in ['male', 'female']:
        all_vertices, all_samples_info = [], []
        seq2info = {
            sbj_id[3:]: dict() for sbj_id in gender2sbj[gender]
        }
        faces = None

        print(f"Loading data for gender: {gender}")
        for sequence in tqdm.tqdm(
            all_seqs_per_gender[gender], total=len(all_seqs_per_gender[gender]), ncols=80
        ):
            seq_info, sbj_params = parse_sequence(sequence)

            seq_info["gender"] = gender
            seq_info["output_seq_path"] = \
                cfg.intercap_sequences / "smplh" / f"{seq_info['sbj_id']}" / \
                f"{seq_info['obj_id']}/{seq_info['action']}"
            seq_info["output_seq_path"].mkdir(exist_ok=True, parents=True)

            obj_id = seq_info["obj_id"]
            sbj_id = seq_info["sbj_id"]
            action = seq_info["action"]

            faces, verts, frame_mask = create_smplx_meshes(seq_info, sbj_params, cfg)

            if not(obj_id in seq2info[sbj_id]):
                seq2info[sbj_id][obj_id] = dict()
            seq2info[sbj_id][obj_id][action] = {
                "output_seq_path": seq_info["output_seq_path"],
                "T": frame_mask.sum()
            }

            seq2info[sbj_id][obj_id][action]['frame_mask'] = frame_mask
            # seq_info["frame_mask"] = frame_mask
            all_vertices.append(verts)
            samples_info = []
            for t in range(seq2info[sbj_id][obj_id][action]["T"]):
                samples_info.append({
                    "sbj_id": sbj_id,
                    "obj_id": obj_id,
                    "action": action,
                    "t": t,
                    "gender": seq_info["gender"],
                    "output_seq_path": seq_info["output_seq_path"],
                })
            all_samples_info.extend(samples_info)

            assert len(samples_info) == len(verts)

        print(f"Converting: {gender}")
        all_vertices = np.concatenate(all_vertices, axis=0)

        mesh_dataset = MeshInMemory(faces, all_vertices, all_samples_info)
        run_smplx2smplh_conversion(mesh_dataset, seq2info, cfg, gender)
