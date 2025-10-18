from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os


# ========================= DATASET ==========================
@dataclass
class DatasetConfig:
    name: str
    root: str
    objects: List[str]
    obj2groupid: Dict[str, int]
    obj2classid:  Dict[str, int]

    # One has to specify either subjects and actions or split_file
    train_subjects: Optional[List[str]]
    train_actions: Optional[List[str]]
    train_split_file: Optional[str]

    # One has to specify either subjects and actions or split_file
    test_subjects: Optional[List[str]]
    test_actions: Optional[List[str]]
    test_split_file: Optional[str]

    downsample_factor: int = 1
    fps_train: int = 10
    fps_eval: int = 1
    augment_rotation: bool = True
    augment_symmetry: bool = True


@dataclass
class BehaveConfig(DatasetConfig):
    # fields should match the fields in HOIDataset class
    name: str = 'behave'
    root: str = os.path.join("${env.datasets_folder}", "behave_smplh")

    objects: List[str] = field(default_factory=lambda: [
        "backpack", "boxlong", "boxtiny", "keyboard", "stool",
        "tablesquare", "yogaball", "basketball", "boxmedium",
        "chairblack", "monitor", "suitcase", "toolbox", "yogamat",
        "boxlarge", "boxsmall", "chairwood", "plasticcontainer",
        "tablesmall", "trashbin"
    ])
    obj2groupid: Dict[str, int] = field(default_factory=lambda: {})
    obj2classid:  Dict[str, int] = field(default_factory=lambda: {})

    # One has to specify either subjects and actions or split_file
    train_subjects: Optional[List[str]] = None
    train_actions: Optional[List[str]] = None
    train_split_file: Optional[str] = os.path.join("${env.assets_folder}", "behave_train.json")

    # One has to specify either subjects and actions or split_file
    test_subjects: Optional[List[str]] = None
    test_actions: Optional[List[str]] = None
    test_split_file: Optional[str] = os.path.join("${env.assets_folder}", "behave_test.json")


@dataclass
class GrabConfig(DatasetConfig):
    # fields should match the fields in HOIDataset class
    name: str = 'grab'
    root: str = os.path.join("${env.datasets_folder}", "grab_smplh")

    objects: List[str] = field(default_factory=lambda: [
        "banana", "binoculars", "camera", "coffeemug",
        "cup", "doorknob", "eyeglasses", "flute",
        "flashlight", "fryingpan", "gamecontroller", "hammer",
        "headphones", "knife", "lightbulb", "mouse",
        "mug", "phone", 'teapot', "toothbrush", "wineglass"
    ])
    obj2groupid: Dict[str, int] = field(default_factory=lambda: {})
    obj2classid:  Dict[str, int] = field(default_factory=lambda: {})

    # One has to specify either subjects and actions or split_file
    train_subjects: Optional[List[str]] = field(default_factory=lambda: [
        "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"
    ])
    train_actions: Optional[List[str]] = None
    train_split_file: Optional[str] = None

    # One has to specify either subjects and actions or split_file
    test_subjects: Optional[List[str]] = field(default_factory=lambda: [
        "s9", "s10"
    ])
    test_actions: Optional[List[str]] = None
    test_split_file: Optional[str] = None


@dataclass
class InterCapConfig(DatasetConfig):
    # fields should match the fields in HOIDataset class
    name: str = 'intercap'
    root: str = os.path.join("${env.datasets_folder}", "intercap_smplh")

    objects: List[str] = field(default_factory=lambda: [
        "ball", "bottle", "briefcase", "chair",
        "cup", "ottoman", "skateboard", "suitcase",
        "tennisracket", "umbrella"
    ])
    obj2groupid: Dict[str, int] = field(default_factory=lambda: {})
    obj2classid: Dict[str, int] = field(default_factory=lambda: {})

    # One has to specify either subjects and actions or split_file
    train_subjects: Optional[List[str]] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08"
    ])
    train_actions: Optional[List[str]] = None
    train_split_file: Optional[str] = None

    # One has to specify either subjects and actions or split_file
    test_subjects: Optional[List[str]] = field(default_factory=lambda: [
        "09", "10"
    ])
    test_actions: Optional[List[str]] = None
    test_split_file: Optional[str] = None

@dataclass
class OmomoConfig(DatasetConfig):
    # fields should match the fields in HOIDataset class
    name: str = 'omomo'
    root: str = os.path.join("${env.datasets_folder}", "omomo_smplh")

    objects: List[str] = field(default_factory=lambda: [
        "clothesstand", "floorlamp", "largebox", "largetable",
        "monitor", "mop", "plasticbox", "smallbox", "smalltable",
        "suitcase", "trashcan", "tripod", "vacuum", "whitechair",
        "woodchair",
    ])
    obj2groupid: Dict[str, int] = field(default_factory=lambda: {})
    obj2classid: Dict[str, int] = field(default_factory=lambda: {})

    # One has to specify either subjects and actions or split_file
    train_subjects: Optional[List[str]] = field(default_factory=lambda: [
        "sub1", "sub2", "sub3", "sub4", "sub5", "sub6", "sub7", "sub8",
        "sub9", "sub10", "sub11", "sub12", "sub13", "sub14", "sub15"
    ])
    train_actions: Optional[List[str]] = None
    train_split_file: Optional[str] = None

    # One has to specify either subjects and actions or split_file
    test_subjects: Optional[List[str]] = field(default_factory=lambda: [
        "sub16", "sub17"
    ])
    test_actions: Optional[List[str]] = None
    test_split_file: Optional[str] = None


@dataclass
class CustomDatasetConfig(DatasetConfig):
    # fields should match the fields in HOIDataset class
    name: str = 'custom'
    root: str = os.path.join("${env.datasets_folder}", "custom_smplh")

    objects: List[str] = field(default_factory=lambda: [
        "chairblack1", "chairblack2", "chairblack3", "chairblack4",
        "chairwood1", "chairwood2", "chairwood3", "chairwood4",
        "monitor1", "stool1", "stool2", "stool3", "stool4",
        "stool5", "stool6"
    ])
    obj2groupid: Dict[str, int] = field(default_factory=lambda: {})
    obj2classid:  Dict[str, int] = field(default_factory=lambda: {})

    # One has to specify either subjects and actions or split_file
    train_subjects: Optional[List[str]] = None
    train_actions: Optional[List[str]] = None
    train_split_file: Optional[str] = None

    # One has to specify either subjects and actions or split_file
    test_subjects: Optional[List[str]] = None
    test_actions: Optional[List[str]] = None
    test_split_file: Optional[str] = os.path.join("${env.assets_folder}", "custom_test.json")
# ============================================================
