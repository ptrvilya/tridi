from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import clip
import h5py
import numpy as np
import torch
import trimesh
import wandb
from trimesh import visual as tr_visual

from config.config import ProjectConfig
from tridi.data import get_eval_dataloader, get_eval_dataloader_random
from tridi.model.base import TriDiModelOutput
from tridi.model.wrappers.contact import ContactModel
from tridi.model.wrappers.mesh import MeshModel
from tridi.utils.geometry import rotation_6d_to_matrix
from tridi.utils.training import TrainState, resume_from_checkpoint

logger = getLogger(__name__)


class Sampler:
    def __init__(self, cfg: ProjectConfig, model):
        self.cfg = cfg

        self.device = torch.device("cuda")
        self.model = model.to(self.device)

        # Resume from checkpoint and create the initial training state
        self.train_state: TrainState = resume_from_checkpoint(cfg, model, None, None)

        # Get dataloaders
        if cfg.sample.dataset == 'random':
            self.dataloaders, self.canonical_obj_meshes, canonical_obj_keypoints\
                = get_eval_dataloader_random(cfg)
        elif cfg.sample.dataset == 'normal':
            self.dataloaders, self.canonical_obj_meshes, canonical_obj_keypoints\
                = get_eval_dataloader(cfg)
        else:
            raise ValueError(f"Unknown dataset type {cfg.sample.dataset}")

        # Model prediction -> meshes
        self.mesh_model = MeshModel(
            model_path=cfg.env.smpl_folder,
            batch_size=2 * cfg.dataloader.batch_size,
            canonical_obj_meshes=self.canonical_obj_meshes,
            canonical_obj_keypoints=canonical_obj_keypoints,
            device=self.device
        )
        self.model.set_mesh_model(self.mesh_model)

        # Create folder for artifacts
        self.base_samples_folder = (Path(self.cfg.run.path) / "artifacts"
                               / f"step_{self.cfg.resume.step}_samples")
        self.base_samples_folder.mkdir(parents=True, exist_ok=True)

        # torch.compile(self.model.denoising_model, mode="reduce-overhead")
        contact_model_type = self.cfg.model_conditioning.use_contacts
        self.contact_model = ContactModel(contact_model_type, device=self.device)
        if len(contact_model_type) > 0 and contact_model_type != "NONE":
            if contact_model_type == "encoder_decimated_clip":
                weights_path = Path(cfg.env.assets_folder) / f"{cfg.model_conditioning.contact_model}.pth"

                if self.cfg.sample.mode[2] == "0":
                    # First set decoder for guidance
                    self.contact_model.set_contact_model(contact_model_type, "model_dec", weights_path)

                    # Then set encoder for conditioning
                    if self.cfg.sample.contacts_mode == "clip":
                        # conditioning on contacts via CLIP
                        enc_dec = "model_enc_clip"
                    elif self.cfg.sample.contacts_mode == "heatmap":
                        # conditioning on contacts via heatmap
                        enc_dec = "model_enc"
                elif self.cfg.sample.mode[2] == "1":
                    # sampling contacts -> we only need the decoder
                    enc_dec = "model_dec"
                else:
                    raise ValueError(f"Unsupported sampling mode {self.cfg.sample.mode} with contacts from {self.cfg.sample.contacts_mode}")
            else:
                raise ValueError(f"Unsupported contact conditioning {self.cfg.model_conditioning.use_contacts}")

            self.contact_model.set_contact_model(contact_model_type, enc_dec, weights_path)
        self.model.set_contact_model(self.contact_model)

    @torch.no_grad()
    def sample_step(self, batch) -> Tuple[TriDiModelOutput, List[str]]:
        self.model.eval()

        # condition on contacts
        captions = []
        if self.cfg.sample.mode[2] == "0": # condition on contacts
            gt_sbj_vertices, gt_sbj_joints = self.mesh_model.get_smpl_th(batch)
            batch.sbj_vertices = gt_sbj_vertices
            batch.sbj_joints = gt_sbj_joints

            if self.cfg.model_conditioning.use_contacts == "encoder_decimated_clip":
                if self.cfg.sample.contacts_mode == 'heatmap':
                    enc_type = "model_enc"
                    obj_name = None
                elif self.cfg.sample.contacts_mode == 'clip':
                    enc_type = "model_enc_clip"
                    obj_name = batch.obj
                else:
                    raise ValueError(f"Unsupported contacts mode {self.cfg.sample.contacts_mode}")

                sbj_contacts, sbj_contacts_full, captions = \
                    self.contact_model.encode_contacts(
                        batch.sbj_vertices, batch.sbj_contact_indexes, batch.obj_keypoints,
                        batch.obj_class, batch.obj_R,
                        enc_type=enc_type, obj_name=obj_name
                    )
                batch.sbj_contacts = sbj_contacts
                batch.sbj_contacts_full = sbj_contacts_full
            elif self.cfg.model_conditioning.use_contacts != "NONE":
                raise ValueError(f"Unsupported contact conditioning {self.cfg.model_conditioning.use_contacts}")

        output = self.model(batch, "sample", sample_type=self.cfg.sample.mode)

        if isinstance(output, tuple):
            output, intermediate_outputs = output

        return self.model.split_output(output), captions

    @staticmethod
    def sample_mode_to_str(sample_mode, contacts_mode):
        sample_str = []
        if sample_mode[0] == "1":
            sample_str.append("sbj")
        if sample_mode[1] == "1":
            sample_str.append("obj")
        if sample_mode[2] == "1":
            sample_str.append("contact")
        elif sample_mode[2] == "0":
            sample_str.append(contacts_mode)
        sample_str = "_".join(sample_str)

        return sample_str

    @torch.no_grad()
    def sample(self):
        # Log general info
        logger.info(
            f'***** Starting sampling *****\n'
            f'    Model: {self.cfg.model_denoising.name}\n'
            f'    Checkpoint step number: {self.cfg.resume.step}\n'
            f'    Number of repetitions: {self.cfg.sample.repetitions}\n'
        )

        for dataloader in self.dataloaders:
            # Log info
            logger.info(
                f'    Sampling mode {self.cfg.sample.mode} for: {dataloader.dataset.name}\n'
                f'    Number of samples: {len(dataloader.dataset)}\n'
            )
            # create folder for the samples
            sample_mode = self.sample_mode_to_str(self.cfg.sample.mode, self.cfg.sample.contacts_mode)
            samples_folder = self.base_samples_folder / f"{dataloader.dataset.name}" / f"{sample_mode}"
            samples_folder.mkdir(parents=True, exist_ok=True)

            for batch_i, batch in enumerate(dataloader):
                for repetition_id in range(self.cfg.sample.repetitions):
                    # Get outputs
                    output, contacts_captions = self.sample_step(batch)

                    # Get contact mask
                    is_sampling_contacts = self.cfg.sample.mode[2] == "1"

                    contacts_mask = self.contact_model.decode_contacts_np(
                        batch.sbj_contacts_full, output.contacts,
                        batch.sbj_contact_indexes, is_sampling_contacts
                    )
                    B = output.contacts.shape[0]
                    contacts_full_color = 128 * np.ones((B, 6890, 4), dtype=np.uint8)
                    contacts_full_color[contacts_mask] = [0, 255, 0, 255]
                    contacts_full_color[:, :, 3] = 255

                    # Convert output to meshes
                    sbj_meshes, obj_meshes = self.mesh_model.get_meshes(
                        output, batch.obj_class, batch.scale, batch.sbj_gender
                    )

                    # Export meshes
                    # For conditional sampling add GT to export
                    for sample_idx in range(len(sbj_meshes)):
                        # save meshes
                        sbj = batch.sbj[sample_idx]
                        obj = batch.obj[sample_idx]
                        act = batch.act[sample_idx]
                        t_stamp = batch.t_stamp[sample_idx]
                        target_folder = samples_folder / sbj / f"{obj}_{act}"
                        target_folder.mkdir(parents=True, exist_ok=True)

                        if self.cfg.sample.mode[0] == "1":
                            target_sbj = f"{t_stamp:04d}_{repetition_id:02d}_subject_sample.ply"
                            save_sbj = True
                        else:
                            target_sbj = f"{t_stamp:04d}_subject_GT.ply"
                            save_sbj = repetition_id == 0

                        if self.cfg.sample.mode[1] == "1":
                            target_obj = f"{t_stamp:04d}_{repetition_id:02d}_object_sample.ply"
                            save_obj = True
                        else:
                            target_obj = f"{t_stamp:04d}_object_GT.ply"
                            save_obj = repetition_id == 0

                        if self.cfg.model_conditioning.use_contacts != "NONE":
                            if self.cfg.sample.mode[2] == "1":
                                target_contact = f"{t_stamp:04d}_{repetition_id:02d}_contact_sample.ply"
                                contact_pc = trimesh.PointCloud(sbj_meshes[sample_idx].vertices)
                                contact_pc.visual = tr_visual.color.VertexColor(contacts_full_color[sample_idx])
                                save_contact = True
                            else:
                                if self.cfg.sample.mode[0] == "1":  # save every iteration
                                    save_contact = True
                                else:  # save only together with ground truth
                                    save_contact = repetition_id == 0

                                if self.cfg.sample.contacts_mode == "heatmap":
                                    if self.cfg.sample.mode[0] == "1":  # save every iteration
                                        target_contact = f"{t_stamp:04d}_{repetition_id:02d}_contact_GT.ply"
                                    else:  # save only together with ground truth
                                        target_contact = f"{t_stamp:04d}_contact_GT.ply"
                                    contact_pc = trimesh.PointCloud(sbj_meshes[sample_idx].vertices)
                                    contact_pc.visual = tr_visual.color.VertexColor(contacts_full_color[sample_idx])
                                else:
                                    target_contact = f"{t_stamp:04d}_contact_GT.txt"
                                    contact_pc = trimesh.PointCloud(sbj_meshes[sample_idx].vertices)
                                    contact_pc.visual = tr_visual.color.VertexColor(contacts_full_color[sample_idx])
                        else:
                            save_contact = False

                        if save_sbj:
                            sbj_meshes[sample_idx].export(target_folder / target_sbj)

                        if save_obj:
                            obj_meshes[sample_idx].export(target_folder / target_obj)

                        if save_contact:
                            if target_contact.endswith("txt"):
                                with open(target_folder / target_contact, "w") as fp:
                                    fp.write(contacts_captions[sample_idx])
                            else:
                                contact_pc.export(target_folder / target_contact)

    @torch.no_grad()
    def sample_to_hdf5(self, target_name="samples.hdf5"):
        # Log general info
        logger.info(
            f'***** Starting sampling *****\n'
            f'    Model: {self.cfg.model_denoising.name}\n'
            f'    Checkpoint step number: {self.cfg.resume.step}\n'
            f'    Number of repetitions: {self.cfg.sample.repetitions}\n'
        )

        for dataloader in self.dataloaders:
            # Log info
            logger.info(
                f'    Sampling mode {self.cfg.sample.mode} for: {dataloader.dataset.name}\n'
                f'    Number of samples: {len(dataloader.dataset)}\n'
            )
            # create folder for the samples
            sample_mode = self.sample_mode_to_str(self.cfg.sample.mode, self.cfg.sample.contacts_mode)
            samples_folder = self.base_samples_folder / f"{dataloader.dataset.name}" / f"{sample_mode}"
            samples_folder.mkdir(parents=True, exist_ok=True)

            # get sequences with lengths from the dataset
            sbj2sct = dataloader.dataset.get_sbj2sct()  # sct = sequence, class_id, T

            base_name = Path(target_name).stem
            S = self.cfg.sample.repetitions
            h5py_files = [h5py.File(str(samples_folder / f"{base_name}_rep_{s:02d}.hdf5"), "w") for s in range(S)]

            for h5py_file in h5py_files:
                for sbj, sequences in sbj2sct.items():
                    sbj_group = h5py_file.create_group(sbj)
                    for seq, class_id, T in sequences:
                        seq_group = sbj_group.create_group(f"{seq}")
                        n_obj_v = self.canonical_obj_meshes[class_id].vertices.shape[0]
                        obj_f = self.canonical_obj_meshes[class_id].faces
                        sbj_f = self.mesh_model.get_faces_np()
                        # old: sbj_vertices, sbj_faces
                        seq_group.create_dataset("sbj_v", shape=(T, 6890, 3))
                        seq_group.create_dataset("sbj_f", shape=(sbj_f.shape[0], 3), data=sbj_f)
                        if self.cfg.model_conditioning.use_contacts != "NONE":
                            seq_group.create_dataset("sbj_contact_z", shape=(T, self.cfg.model.data_contact_channels))
                            seq_group.create_dataset("sbj_contact", shape=(T, 6890))
                        # old: obj_vertices, obj_faces
                        seq_group.create_dataset("obj_v", shape=(T, n_obj_v, 3))
                        seq_group.create_dataset("obj_f", shape=(obj_f.shape[0], 3), data=obj_f)
                        # sbj params, old: sbj_pose, sbj_c, sbj_shape
                        seq_group.create_dataset("sbj_smpl_pose", shape=(T, 1+21+15+15, 9))
                        seq_group.create_dataset("sbj_smpl_transl", shape=(T, 3))
                        seq_group.create_dataset("sbj_smpl_betas", shape=(T, 10))
                        seq_group.create_dataset("sbj_j", shape=(T, 73, 3))
                        # obj params
                        seq_group.create_dataset("obj_c", shape=(T, 3))
                        seq_group.create_dataset("obj_R", shape=(T, 9))
                        # attributes
                        seq_group.attrs['T'] = T


            # prediction loop
            for batch_idx, batch in enumerate(dataloader):
                for repetition_id in range(self.cfg.sample.repetitions):
                    # Get outputs
                    output, captions = self.sample_step(batch)
                    sbj_vertices, obj_vertices, sbj_joints = self.mesh_model.get_meshes_th(
                        output, batch.obj_class, batch.scale, sbj_gender=batch.sbj_gender, return_joints=True
                    )

                    # convert rotation from 6d to matrix
                    # output = output
                    sbj_pose = torch.cat([
                        output.sbj_global,
                        output.sbj_pose,
                    ], dim=1).reshape(-1, 52, 6)
                    sbj_pose = rotation_6d_to_matrix(sbj_pose).reshape(-1, 52, 9).cpu().numpy()
                    obj_R = output.obj_R.reshape(-1, 1, 6)
                    obj_R = rotation_6d_to_matrix(obj_R).reshape(-1, 9).cpu().numpy()

                    # Decode contacts
                    if self.cfg.model_conditioning.use_contacts != "NONE":
                        is_sampling_contacts = self.cfg.sample.mode[2] == "1"
                        contacts_mask = self.contact_model.decode_contacts_np(
                            batch.sbj_contacts_full, output.contacts,
                            batch.sbj_contact_indexes, is_sampling_contacts
                        )
                        if is_sampling_contacts:
                            contacts_z = output.contacts.cpu().numpy()
                        else:
                            contacts_z = batch.sbj_contacts.cpu().numpy()

                    # save to hdf5
                    sbj_vertices, obj_vertices = sbj_vertices.cpu().numpy(), obj_vertices.cpu().numpy()
                    for sample_idx, class_id in enumerate(batch.obj_class.cpu().numpy()):
                        sbj = batch.sbj[sample_idx]
                        obj = batch.obj[sample_idx]
                        act = batch.act[sample_idx]
                        t_stamp = batch.t_stamp[sample_idx].item()

                        sbj_group = h5py_files[repetition_id][sbj]
                        seq_group = sbj_group[f"{obj}_{act}"]
                        n_obj_v = self.canonical_obj_meshes[class_id].vertices.shape[0]

                        seq_group['sbj_v'][t_stamp] = sbj_vertices[sample_idx]
                        seq_group['obj_v'][t_stamp] = obj_vertices[sample_idx][:n_obj_v]
                        if self.cfg.model_conditioning.use_contacts != "NONE":
                            seq_group['sbj_contact_z'][t_stamp] = contacts_z[sample_idx]
                            seq_group['sbj_contact'][t_stamp] = contacts_mask[sample_idx]
                        seq_group['sbj_smpl_pose'][t_stamp] = sbj_pose[sample_idx]
                        seq_group['sbj_smpl_transl'][t_stamp] = output.sbj_c[sample_idx].cpu().numpy()
                        seq_group['sbj_smpl_betas'][t_stamp] = output.sbj_shape[sample_idx].cpu().numpy()
                        seq_group['sbj_j'][t_stamp] = sbj_joints[sample_idx].cpu().numpy()
                        seq_group['obj_c'][t_stamp] = output.obj_c[sample_idx].cpu().numpy()
                        seq_group['obj_R'][t_stamp] = obj_R[sample_idx]
