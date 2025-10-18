from dataclasses import dataclass
import pickle as pkl
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import List, Optional, NamedTuple, Dict

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf

from .batch_data import BatchData

logger = getLogger(__name__)


class DataSample(NamedTuple):
    object: str
    t_stamp: int
    obj_class: int
    obj_group: int


@dataclass
class RandomDataset:
    name: str
    root: Path
    num_samples: int
    class_distribution: str
    objects: List[str]
    obj2classid: Dict[str, int]
    obj2groupid: Dict[str, int]
    include_contacts: str = ""
    include_pointnext: bool = False
    assets_folder: Optional[Path] = None
    # canonical_obj_meshes: Dict[int, trimesh.Trimesh]
    # classid2obj: List[str]

    def __post_init__(self) -> None:
        # convert to dict
        self.obj2classid = OmegaConf.to_container(self.obj2classid)

        # Create reverse mapping
        self.classid2obj = {i: obj for obj, i in self.obj2classid.items()}

        # Determine number of classes
        unique_classes = np.sort(np.unique(list(self.obj2classid.values())))
        num_classes = len(unique_classes)

        # Sample target class
        if self.class_distribution == "equal":
            sampled_classes = np.repeat(unique_classes, self.num_samples // num_classes)
            self.sampled_classes = np.pad(sampled_classes, (0, self.num_samples % num_classes), mode="edge")
        elif self.class_distribution == "uniform":
            self.sampled_classes = np.random.choice(unique_classes, size=self.num_samples)
        else:
            raise ValueError(f"Unknown class distribution {self.class_distribution}")

        self.data = []
        classes, counts = np.unique(self.sampled_classes, return_counts=True)
        for class_id, count in zip(classes, counts):
            self.data.extend([
                DataSample(
                    object=self.classid2obj[class_id],
                    t_stamp=t,
                    obj_class=class_id,
                    obj_group=self.obj2groupid[self.classid2obj[class_id]]
                ) for t in range(count)
            ])

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

        if self.include_contacts == "encoder_decimated_clip":
            self.sbj_contact_indexes = torch.from_numpy(
                np.load(self.assets_folder / "smpl_template_decimated_idxs.npy")
            )
        else:
            self.sbj_contact_indexes = None

        # Log info
        logger.info(self.__str__())

    def get_sbj2sct(self):
        nsamples_per_class = self.get_nsamples_per_class()
        return {
            "Sbj00": [(f"{self.classid2obj[class_id]}_", class_id, T) for class_id, T in nsamples_per_class.items()]
        }

    def get_nsamples_per_class(self):
        classes, counts = np.unique(self.sampled_classes, return_counts=True)
        return dict(zip(classes, counts))

    def __str__(self) -> str:
        return f"RandomDataset class distribution={self.class_distribution} #frames={len(self.sampled_classes)}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> BatchData:
        # Get sampled class
        sampled_data = self.data[idx]

        if self.include_pointnext:
            obj_pointnext = torch.tensor(self.obj_pointnext[sampled_data.obj_class], dtype=torch.float)
        else:
            obj_pointnext = None

        # Fill BatchData isntance
        batch_data = BatchData(
            sbj="Sbj00",
            act="",
            obj=sampled_data.object,
            obj_class=torch.tensor(sampled_data.obj_class, dtype=torch.long),
            obj_group=torch.tensor(sampled_data.obj_group, dtype=torch.long),
            t_stamp=sampled_data.t_stamp,
            # conditioning
            obj_pointnext=obj_pointnext,
            sbj_contact_indexes=self.sbj_contact_indexes,
        )

        return batch_data

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

    def _load_obj_pointnext(self) -> Dict[int, np.ndarray]:
        with (self.root / "object_pointnext.pkl").open("rb") as fp:
            _obj_pointnext = pkl.load(fp)

        obj_pointnext = {self.obj2classid[k]: v for k, v in _obj_pointnext.items()}

        return obj_pointnext
