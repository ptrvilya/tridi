import json
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import List, Optional, NamedTuple, Dict

import h5py
import numpy as np
import torch
import trimesh

from .batch_data import BatchData
from ..utils.geometry import matrix_to_rotation_6d

logger = getLogger(__name__)


class H5DataSample(NamedTuple):
    subject: str
    object: str
    action: str
    name: str
    t_stamp: int
    obj_class: int
    obj_group: int


@dataclass
class HOIDataset:
    # dataset class for GRAB, BEHAVE, InterCap, OMOMO
    name: str
    root: Path
    split: str
    objects: List[str]
    obj2classid: Dict[str, int]
    obj2groupid: Dict[str, int]
    downsample_factor: int = 1
    h5dataset_path: Path = None
    preload_data: bool = True
    subjects: Optional[List[str]] = field(default_factory=list)
    actions: Optional[List[str]] = field(default_factory=list)
    split_file: Optional[str] = None
    behave_repeat_fix: bool = False  # repeating the data for classes with only 1 fps annotations
    augment_rotation: bool = False
    augment_symmetry: bool = False
    use_relative_obj_c: bool = False
    include_contacts: str = ""
    include_pointnext: bool = False
    assets_folder: Optional[Path] = None
    fps: Optional[int] = 30
    # h5dataset: h5py.File
    # sbj2objact: Dict[str, List[(str, str)]]
    # data: List[H5DataSample]  
    # canonical_obj_meshes: Dict[int, trimesh.Trimesh]
    # canonical_obj_keypoints: Dict[int, Dict[str, np.ndarray]]

    def __post_init__(self) -> None:
        # Open h5 dataset
        self.h5dataset_path = self.root / f"dataset_{self.split}_{self.fps}fps.hdf5"
        if self.preload_data:
            self.h5dataset = self._preload_h5_dataset(self.h5dataset_path)
            logger.info("Preloaded H5 dataset into memory.")
        else:
            self.h5dataset = h5py.File(self.h5dataset_path, "r")

        # Create reverse mapping
        self.classid2obj = {i: obj for obj, i in self.obj2classid.items()}
        
        # Get mapping from subject to object-action pairs
        if self.split_file is not None:
            self.sbj2objact = self._get_sbj2objact_from_split()  # assumes file with sbj, obj, act triplets
        else:
            self.sbj2objact = self._get_sbj2objact()  # assumes pre-defined subject split

        # Load H5DataSample and sort by timestamp
        self.data = self._load_data()
        self._sort_data()

        # Load canonical object meshes
        self.canonical_obj_meshes = self._load_canonical_meshes()
        self.canonical_obj_keypoints = self._load_canonical_keypoints()

        # Load pointnext object encoding
        if self.include_pointnext:
            self.obj_pointnext = self._load_obj_pointnext()
        else:
            self.obj_pointnext = None

        # Load contact indeces
        if self.assets_folder is None:
            self.assets_folder = Path("./assets")

        if self.include_contacts == "encoder_decimated_clip" or \
                self.include_contacts == "NONE":
            self.sbj_contact_indexes = torch.from_numpy(
                np.load(self.assets_folder / "smpl_template_decimated_idxs.npy")
            )
        else:
            self.sbj_contact_indexes = None

        # Log info
        logger.info(self.__str__())
    
    def __str__(self) -> str:
        return f"HOIDataset {self.name}: split={self.split} #frames={len(self.data)}"

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _preload_h5_dataset(h5dataset_path: Path):
        data_dict = dict()
        with h5py.File(h5dataset_path, "r") as h5_dataset:
            for sbj in h5_dataset.keys():
                data_dict[sbj] = dict()
                for obj_act in h5_dataset[sbj].keys():
                    data_dict[sbj][obj_act] = dict()
                    for key in h5_dataset[sbj][obj_act].keys():
                        data_dict[sbj][obj_act][key] = h5_dataset[sbj][obj_act][key][:]
                    # copy attributes
                    data_dict[sbj][obj_act]["_attrs"] = dict(h5_dataset[sbj][obj_act].attrs)
        return data_dict

    def get_sequence(self, sbj: str, obj: str, act: str) -> h5py.Group:
        return self.h5dataset[sbj][f"{obj}_{act}"]

    @staticmethod
    def _apply_z_rotation_augmentation(sbj_global, obj_v, obj_R, obj_c):
        # Z - rotation augmentation
        angle = np.random.choice(np.arange(-np.pi, np.pi, np.pi / 36))

        # Rotation matrix along z axis
        R_aug_z = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ], dtype=np.float32)

        obj_v = np.dot(R_aug_z, obj_v.T).T
        sbj_global = np.dot(R_aug_z, sbj_global.reshape(3, 3))
        obj_R = np.dot(R_aug_z, obj_R.reshape(3, 3))
        obj_c = np.dot(R_aug_z, obj_c.reshape(3, 1)).reshape(3)

        return sbj_global, obj_v, obj_R, obj_c

    @staticmethod
    def _apply_symmetry_augmentation(body_model_params, obj_R, obj_c):
        # symmetrical mapping for body joints
        body_sym_map = np.array(
            [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13, 12, 14, 16, 15, 18, 17, 20, 19]
        )
        body_model_params["body_pose"] = body_model_params["body_pose"][body_sym_map]

        # z y x -> -z -y x
        sign_flip = np.array([[
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0]
        ]], dtype=np.float32)

        # flip rotations
        # IMPORTANT: FLIP FIXER
        body_model_params = {k: v * sign_flip if k != "global_orient" else v for k, v in body_model_params.items()}
        obj_R *= sign_flip

        # Translation is the same, but the component orthogonal to the plane (X-axis) must be inverted
        obj_c[0] *= -1

        # flipping left and right hands in the output
        lh, rh = body_model_params["left_hand_pose"], body_model_params["right_hand_pose"]
        body_model_params["left_hand_pose"] = rh
        body_model_params["right_hand_pose"] = lh

        return body_model_params, obj_R, obj_c

    def __getitem__(self, idx: int) -> BatchData:
        sample = self.data[idx]
        sequence = self.get_sequence(sample.subject, sample.object, sample.action)

        # parse attrs
        if self.preload_data:
            sbj_gender = sequence["_attrs"]['gender']
        else:
            sbj_gender = sequence.attrs['gender']

        # object
        obj_R = sequence['obj_R'][sample.t_stamp].reshape(3, 3)
        obj_c = sequence['obj_c'][sample.t_stamp]

        # ==> augmentations
        if self.augment_symmetry and np.random.rand() > 0.5:
            body_model_params = {
                "global_orient": sequence['sbj_smpl_global'][sample.t_stamp].reshape(1, 3, 3),
                "body_pose": sequence['sbj_smpl_body'][sample.t_stamp].reshape(-1, 3, 3),
                "left_hand_pose":  sequence['sbj_smpl_lh'][sample.t_stamp].reshape(-1, 3, 3),
                "right_hand_pose": sequence['sbj_smpl_rh'][sample.t_stamp].reshape(-1, 3, 3)
            }

            # perfrom horizontal flip
            body_model_params, obj_R, obj_c = self._apply_symmetry_augmentation(
                body_model_params, obj_R.reshape(1, 3, 3), obj_c
            )

            # save subject params
            sbj_pose = np.concatenate([
                body_model_params['body_pose'],
                body_model_params['left_hand_pose'],
                body_model_params['right_hand_pose']
            ], axis=0).reshape((51, 3, 3))
            sbj_global = body_model_params['global_orient'][0].reshape((3, 3))
        else:
            sbj_pose = np.concatenate([
                sequence['sbj_smpl_body'][sample.t_stamp],
                sequence['sbj_smpl_lh'][sample.t_stamp],
                sequence['sbj_smpl_rh'][sample.t_stamp],
            ], axis=0).reshape((51, 3, 3))
            sbj_global = sequence['sbj_smpl_global'][sample.t_stamp]

        # pose canonical keypoints for object
        obj_keypoints_can = np.copy(self.canonical_obj_keypoints[sample.obj_class]["cartesian"])
        obj_keypoints_gt = np.dot(obj_R.reshape(3, 3), obj_keypoints_can.T).T + obj_c.reshape(1, 3)

        sbj_global = sbj_global.reshape(3, 3)
        obj_R = obj_R.reshape(3, 3)
        obj_c = obj_c.reshape(3)
        if self.augment_rotation and np.random.rand() > 0.25:
            sbj_global, obj_keypoints_gt, obj_R, obj_c = self._apply_z_rotation_augmentation(
                sbj_global, obj_keypoints_gt, obj_R, obj_c
            )
        # <==
        obj_v = torch.tensor(obj_keypoints_gt.reshape(-1, 3), dtype=torch.float)

        # convert to 6d representation
        sbj_global = matrix_to_rotation_6d(sbj_global.reshape(3, 3)).reshape(-1)
        sbj_pose = matrix_to_rotation_6d(sbj_pose).reshape(-1)
        obj_R = matrix_to_rotation_6d(obj_R)

        # Conditioning
        if self.include_pointnext:
            obj_pointnext = torch.tensor(self.obj_pointnext[sample.obj_class], dtype=torch.float)
        else:
            obj_pointnext = None

        # Fill BatchData isntance
        batch_data = BatchData(
            # metadata
            meta={
                "name": sample.name,
                "t_stamp": sample.t_stamp,
            },
            sbj=sample.subject,
            obj=sample.object,
            act=sample.action,
            t_stamp=sample.t_stamp,
            # subject
            sbj_shape=torch.tensor(sequence['sbj_smpl_betas'][sample.t_stamp], dtype=torch.float),
            sbj_global=sbj_global,
            sbj_pose=sbj_pose,
            sbj_c=torch.tensor(sequence['sbj_smpl_transl'][sample.t_stamp], dtype=torch.float),
            sbj_gender=torch.tensor(sbj_gender == 'male', dtype=torch.bool),
            # object
            obj_R=obj_R,
            obj_c=obj_c,
            obj_class=torch.tensor(sample.obj_class, dtype=torch.long),
            obj_group=torch.tensor(sample.obj_group, dtype=torch.long),
            obj_keypoints=obj_v,  # actually obj_keypoints
            # conditioning
            obj_pointnext=obj_pointnext,
            # Contacts
            sbj_contact_indexes=self.sbj_contact_indexes,
            sbj_contacts=None,  # computed afterward
            # preprocessing
            scale=torch.tensor(sequence['prep_s'][sample.t_stamp], dtype=torch.float)
        )

        return batch_data

    def get_sbj2sct(self):
        # used to create hdf5 file for sampling
        sbj2sct = defaultdict(list)

        for sbj, obj_acts in self.sbj2objact.items():
            for (obj, act) in obj_acts:
                seq = self.get_sequence(sbj, obj, act)
                if self.preload_data:
                    T = seq["_attrs"]["T"]
                else:
                    T = seq.attrs["T"]

                sbj2sct[sbj].append((f"{obj}_{act}", self.obj2classid[obj], T))
        return sbj2sct

    def _get_sbj2objact_from_split(self) -> Dict[str, List[str]]:
        with open(self.split_file, "r") as fp:
            split = json.load(fp)

        subject2objact = defaultdict(list)
        for sbj, obj_act in split:
            _oa_split = obj_act.split("_")
            obj = _oa_split[0]
            act = "_".join(_oa_split[1:])

            if self.actions is None or (len(self.actions) > 0 and act in self.actions):
                if self.objects is None or (len(self.objects) > 0 and obj in self.objects):
                    subject2objact[sbj].append((obj, act))

        return subject2objact

    def _get_sbj2objact(self) -> Dict[str, List[str]]:
        subject2objact = defaultdict(list)

        assert self.subjects is not None, "Subjects for the dataset are not specified"
        assert len(self.subjects) > 0, "Subjects for the dataset are not specified"

        for sbj in self.subjects:
            obj_acts = list(self.h5dataset[sbj].keys())
            for obj_act in obj_acts:
                obj_act = str(obj_act).split("_")
                obj = obj_act[0]
                act = "_".join(obj_act[1:])

                if self.actions is None or (len(self.actions) > 0 and act in self.actions):
                    if self.objects is None or (len(self.objects) > 0 and obj in self.objects):
                        subject2objact[sbj].append((obj, act))

        return subject2objact
    
    def _load_data(self) -> List[H5DataSample]:
        logger.info(f"HOIDataset {self.name}: loading from {self.h5dataset_path}.")

        data = []
        T = 0
        for sbj, obj_acts in self.sbj2objact.items():
            for (obj, act) in obj_acts:
                seq = self.get_sequence(sbj, obj, act)
                if self.preload_data:
                    t_stamps = list(range(seq["_attrs"]["T"]))
                else:
                    t_stamps = list(range(seq.attrs["T"]))

                if self.behave_repeat_fix and obj in ["basketball", "keyboard"]:
                    t_stamps = t_stamps * 15  # repeating the data

                T += len(t_stamps)
                seq_data = [
                    H5DataSample(
                        subject=sbj,
                        object=obj,
                        action=act,
                        name=f"{sbj}_{obj}_{act}",
                        t_stamp=t_stamp,
                        obj_class=self.obj2classid[obj],
                        obj_group=self.obj2groupid[obj]
                    ) for t_stamp in t_stamps
                ]

                if self.downsample_factor > 1:
                    seq_data = seq_data[::self.downsample_factor]
                data.extend(seq_data)
        logger.info(f"HOI dataset {self.name} {self.split} has {T} frames.")
        return data

    def _sort_data(self) -> None:
        self.data = sorted(
            self.data,
            key=lambda f: (
                f.name,
                f.t_stamp or 0,
            ),
        )

    def _load_obj_pointnext(self) -> Dict[int, np.ndarray]:
        with (self.root / "object_pointnext.pkl").open("rb") as fp:
            _obj_pointnext = pkl.load(fp)

        obj_pointnext = {self.obj2classid[k]: v for k, v in _obj_pointnext.items()}

        return obj_pointnext

    def _load_canonical_meshes(self) -> Dict[int, trimesh.Trimesh]:
        canonical_obj_meshes = dict()
        for class_id, object_name in self.classid2obj.items():
            canonical_obj_meshes[class_id] = trimesh.load(
                str(self.root / "object_meshes" / f"{object_name}.ply"),
                process=False
            )
        return canonical_obj_meshes

    def _load_canonical_keypoints(self) -> Dict[int, Dict[str, np.ndarray]]:
        canonical_obj_keypoints = dict()
        for class_id, object_name in self.classid2obj.items():
            canonical_obj_keypoints[class_id] = dict(np.load(
                str(self.root / "object_keypoints" / f"{object_name}.npz")
            ))
        return canonical_obj_keypoints
