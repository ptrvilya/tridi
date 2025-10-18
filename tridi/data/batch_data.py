from dataclasses import dataclass, field, fields
from typing import List, Optional, Union

import torch
from torch import Tensor


@dataclass
class BatchData:
    # info
    sbj: Union[str, List[str], None]
    path: Union[str, List[str], None] = None
    act: Union[str, List[str], None] = None
    obj: Union[str, List[str], None] = None
    t_stamp: Union[int, List[int], None] = None
    # sbj
    sbj_shape: Optional[Tensor] = None  # dim is 10
    sbj_global: Optional[Tensor] = None  # R is 6d
    sbj_pose: Optional[Tensor] = None  # R is 6d => dim is 51*6
    sbj_c: Optional[Tensor] = None  # dim is 3
    sbj_vertices: Optional[Tensor] = None  # dim is 6890x3
    sbj_joints: Optional[Tensor] = None  # dim is 6890x3
    sbj_gender: Optional[Tensor] = None  # dim is 1
    # sbj-obj contacts
    sbj_contact_indexes: Optional[Tensor] = None  # dim is 256
    sbj_contacts: Optional[Tensor] = None  # dim is 256
    sbj_contacts_full: Optional[Tensor] = None  # dim is 256
    # obj
    obj_R: Optional[Tensor] = None  # R is 6d
    obj_c: Optional[Tensor] = None  # dim is 3d
    obj_can_normals: Optional[Tensor] = None  # dim is 1500x3
    # obj_pose: Optional[Tensor] = None  # R is 6d, t is 3d => dim is 9d
    obj_keypoints: Optional[Tensor] = None  # dim is n_points x 3
    # conditioning
    obj_class: Optional[Tensor] = None  # dim is 1
    obj_group: Optional[Tensor] = None  # dim is 1
    obj_pointnext: Optional[Tensor] = None  # dim is 1024
    # additional data
    scale: Optional[Tensor] = None

    meta: dict = field(default_factory=lambda: {})

    def to(self, *args, **kwargs):
        new_params = {}
        for field_name in iter(self):
            value = getattr(self, field_name)
            if isinstance(value, (torch.Tensor)):
                new_params[field_name] = value.to(*args, **kwargs)
            else:
                new_params[field_name] = value
        batch_data = type(self)(**new_params)
        return batch_data

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    def batch_size(self):
        for f in iter(self):
            if f != "meta":
                attr = self.__getattribute__(f)
                if not (attr is None):
                    return len(attr)

    # the following functions make sure **batch_data can be passed to functions
    def __iter__(self):
        for f in fields(self):
            if f.name.startswith("_"):
                continue

            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return sum(1 for f in iter(self))

    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """

        elem = batch[0]

        if isinstance(elem, cls):
            collated = {}
            for f in fields(elem):
                if not f.init:
                    continue

                list_values = [getattr(d, f.name) for d in batch]
                
                collated[f.name] = (
                    cls.collate(list_values)
                    if all(list_value is not None for list_value in list_values)
                    else None
                )
            return cls(**collated)
        else:
            return torch.utils.data._utils.collate.default_collate(batch)
