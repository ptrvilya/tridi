import argparse
from pathlib import Path
from types import SimpleNamespace
import pickle as pkl

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


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return npz


def prepare_params(params, frame_mask, dtype = np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}


def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


class MeshInMemory(Dataset):
    def __init__(self, faces, verts, seq_info):
        self.faces = faces
        self.verts = verts
        self.seq_info = seq_info
        self.num_items = len(self.verts)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index):
        mesh_verts = self.verts[index]

        return {
            'vertices': np.asarray(mesh_verts, dtype=np.float32),
            'faces': np.copy(self.faces),
            'indices': index,
            'paths': f"{self.seq_info['sbj_id']}/"
                     f"{self.seq_info['obj_name']}_{self.seq_info['action']}/"
                     f"{index:04d}.obj"
        }


def create_smplx_meshes(seq_data, cfg):
    n_comps = seq_data['n_comps']
    frame_mask = filter_contact_frames(cfg, seq_data)

    if cfg.downsample:
        downsample_mask = np.zeros_like(frame_mask)
        downsample_mask[::4] = True
        frame_mask = np.logical_and(frame_mask, downsample_mask)

    T = frame_mask.sum()

    sbj_params = prepare_params(seq_data["body"]["params"], frame_mask)

    # create SMPL-X mesh
    sbj_mesh = str(cfg.grab_path / seq_data["body"]["vtemp"])  # template subject mesh
    sbj_mesh = trimesh.load(sbj_mesh, process=False)
    sbj_vtemp = np.array(sbj_mesh.vertices)
    sbj_model = smplx.create(model_path=str(cfg.model_path), model_type='smplx', gender=seq_data['gender'],
                             num_pca_comps=n_comps, v_template=sbj_vtemp, batch_size=T)
    sbj_parms = params2torch(sbj_params)
    # dict_keys(['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
    # 'left_hand_pose', 'right_hand_pose', 'fullpose', 'expression'])
    sbj_output = sbj_model(**sbj_parms)
    sbj_verts = to_cpu(sbj_output.vertices)

    return frame_mask, sbj_mesh.faces, sbj_verts


def run_smplx2smplh_conversion(dataset, seq_info, cfg):
    exp_conf = {
        "deformation_transfer_path": "./transfer_data/smplx2smplh_deftrafo_setup.pkl",
        "mask_ids_fname": '',
        "summary_steps": 200,

        "edge_fitting": {"per_part": False},

        "optim": {"type": 'lbfgs', "maxiters": 200, "gtol": 1e-07},
        "batch_size": cfg.batch_size,

        "body_model": {
            "model_type": "smplh",
            # SMPL+H has no neutral model, so we have to manually select the gender
            "gender": seq_info["gender"],
            "ext": 'pkl',
            "folder": cfg.model_path,
            "use_compressed": False,
            "use_face_contour": True,
            "smplh": {"betas": {"num": 10}}
        }
    }

    device = torch.device("cuda:0")
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
    exp_omegaconf = omegaconf_from_dict(exp_conf)

    model_path = exp_omegaconf.body_model.folder
    body_model = build_layer(model_path, **exp_omegaconf.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)
    mask_ids = None

    deformation_transfer_path = exp_omegaconf.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    body = {
        'transl': [], 'global_orient': [], 'body_pose': [], 'betas': [], 'left_hand_pose': [], 'right_hand_pose': [],
        'full_pose': []
    }

    sbj_model = smplx.build_layer(
        model_path=str(cfg.model_path), model_type="smplh", gender=seq_data['gender'], 
        num_betas=10, batch_size=cfg.batch_size, num_pca_comps=12, use_pca=False, use_compressed=False
    ).to(device)

    for batch_index, batch in enumerate(tqdm.tqdm(dataloader)):
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)
        var_dict = run_fitting(exp_omegaconf, batch, body_model, def_matrix, mask_ids)

        indexes = batch['indices'].detach().cpu().numpy().tolist()
        vertices = var_dict.pop("vertices")
        faces = var_dict.pop("faces")
        var_dict.pop("v_shaped"), var_dict.pop("joints")

        # optionally save meshes
        if cfg.save_meshes:
            for i, index in enumerate(indexes):
                output_mesh_path = str(seq_info["output_seq_path"] / f"{index:04d}.obj")
                mesh_smplh = np_mesh_to_o3d(vertices[i], faces)
                o3d.io.write_triangle_mesh(output_mesh_path, mesh_smplh)
                
                output_mesh_path = str(seq_info["output_seq_path"] / f"smplx_{index:04d}.obj")
                v = batch['vertices'][i].detach().cpu().numpy()
                f = batch['faces'][i].detach().cpu().numpy()
                mesh_smplx = np_mesh_to_o3d(v, f)
                o3d.io.write_triangle_mesh(output_mesh_path, mesh_smplx)

        # save parameters per-batch
        for k in body.keys():
            body[k].append(var_dict[k].detach().cpu())

    # concatenate per-batch
    for k in body.keys():
        body[k] = torch.cat(body[k], dim=0)

    # convert pose matrices to rotvec
    matrices_blhrh = [body["body_pose"], body["left_hand_pose"], body["right_hand_pose"]]
    matrices_blhrh = torch.cat(matrices_blhrh, dim=1).numpy()
    rotvec_blhrh = np.zeros(matrices_blhrh.shape[:3], dtype=np.float32)
    for t in range(matrices_blhrh.shape[0]):
        R = Rotation.from_matrix(matrices_blhrh[t])
        rotvec_blhrh[t] = R.as_rotvec()
        
    body["pose_blhrh_rotvec"] = rotvec_blhrh
    
    # save converted data
    sequence_data_path = seq_info.pop("output_seq_path") / f"sequence_data.pkl"
    seq_info["body"] = body
    with sequence_data_path.open('wb') as fp:
        pkl.dump(seq_info, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conversion of GRAB annotations from SMPL-X to SMPL+H.')

    parser.add_argument('-g', "--grab-path", type=Path)
    parser.add_argument('-s', "--smplx-path", type=Path)

    args = parser.parse_args()

    # Adjust parameters manually
    cfg = {
        'only_contact': True,
        'grab_path': args.grab_path,

        'batch_size': 1024,
        'num_workers': 6,

        'downsample': True,  # downsample from 120 fps to 30 fps
        'save_meshes': False,
        
        'objects': [
            "banana", "binoculars", "camera", "coffeemug",
            "cup", "doorknob", "eyeglasses", "flute", 
            "flashlight", "fryingpan", "gamecontroller", "hammer",
            "headphones", "knife", "lightbulb", "mouse",
            "mug", "phone", 'teapot', "toothbrush", "wineglass"
        ],

        # body and hand model path
        'model_path': str(args.smplx_path),
        'transfer_data_path': "./transfer_data",
    }

    cfg = SimpleNamespace(**cfg)
    _all_seqs = list(cfg.grab_path.glob('grab/*/*.npz'))
    
    all_seqs = []
    if len(cfg.objects) > 0:
        for seq in _all_seqs:
            if seq.stem.split("_")[0] in cfg.objects:
                all_seqs.append(seq)
    else:
        all_seqs = _all_seqs

    for sequence in tqdm.tqdm(all_seqs, total=len(all_seqs), ncols=80):
        seq_data = parse_npz(sequence)
        seq_info = {
            "gender": seq_data['gender'],
            "sbj_id": str(seq_data["sbj_id"]),
            "action": "_".join(sequence.stem.split("_")[1:]),
            "obj_name": str(seq_data["obj_name"]),
        }
        seq_info["output_seq_path"] = \
            cfg.grab_path / "grab_smplh" / f"{seq_info['sbj_id']}" / \
            f"{seq_info['obj_name']}_{seq_info['action']}"
        seq_info["output_seq_path"].mkdir(exist_ok=True, parents=True)

        frame_mask, faces, verts = create_smplx_meshes(seq_data, cfg)
        seq_info["frame_mask"] = frame_mask

        mesh_dataset = MeshInMemory(faces, verts, seq_info)
        run_smplx2smplh_conversion(mesh_dataset, seq_info, cfg)
