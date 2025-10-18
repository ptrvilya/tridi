from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os

from .environment import EnvironmentConfig


# ========================== Preprocessing configs =============================
@dataclass
class PreprocessBehaveConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "behave/")
    target: str = os.path.join("${env.datasets_folder}", "behave_smplh/")
    orig_objects_path: str = os.path.join("${behave.root}", "objects")
    can_objects_path: str = os.path.join("${behave.root}", "canonicalized_objects")
    full_anno_path: str = os.path.join("${behave.root}", "behave-30fps-params-v1")

    # Split (needed to correctly set resulting filename and to load sequences from split file)
    split: str = "train"  # "train", "test"
    # Path to a split_file (for selecting sequences based on split)
    split_file: str = os.path.join("${behave.root}", "split.json")
    # Useful to preprocess sequences for basketball and keyboard that only have 1fps annotations
    # together with 30fps sequences
    combine_1fps_and_30fps: bool = True
    # Downsample from to 10fps or 1fps, only applicable to 30fps data
    downsample: str = "10fps"
    # Preprocess only selected subjects
    subjects: List = field(default_factory=lambda: ["*"])
    # Preprocess only selected objects
    objects: List = field(default_factory=lambda: [])
    # Threshold for computing human-object contacts (if no precomputed contacts are provided)
    contact_threshold: float = 0.020
    # Generate object keypoints after preprocessing
    generate_obj_keypoints: bool = False


@dataclass
class PreprocessGrabConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "grab/")
    target: str = os.path.join("${env.datasets_folder}", "grab_smplh/")
    objects_path: str = os.path.join("${grab.root}", "tools/object_meshes/decimated_meshes")

    # Split (needed to correctly set resulting filename)
    split: str = "train"  # "train", "test"
    # Preprocess only frames with human-object contacts
    only_contact_frames: bool = True
    # Downsample from 120fps to 30fps or 10fps
    downsample: str = "10fps" # "None", "30fps", "10fps", "1fps"
    # Use modified meshes with fewer vertices
    use_decimated_obj_meshes: bool = True
    # Preprocess only selected subjects
    subjects: List = field(default_factory=lambda: ["*"])
    # Preprocess only selected objects
    objects: List = field(default_factory=lambda: [
        "banana", "binoculars", "camera", "coffeemug",
        "cup", "doorknob", "eyeglasses", "flute",
        "flashlight", "fryingpan", "gamecontroller", "hammer",
        "headphones", "knife", "lightbulb", "mouse",
        "mug", "phone", 'teapot', "toothbrush", "wineglass"
    ])
    # Load generated SMPL-h meshes (works only with SMPL-h)
    load_existing_sbj_meshes: bool = False
    # Generate object keypoints after preprocessing
    generate_obj_keypoints: bool = False


@dataclass
class PreprocessIntercapConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "intercap/")
    target: str = os.path.join("${env.datasets_folder}", "intercap_smplh/")
    objects_path: str = os.path.join("${intercap.root}", "tools/canonicalized_objects")

    # Split (needed to use pre-defined splits)
    split: str = "train"  # "train", "test"
    # Use manualy canonicalized meshes (False is not supported)
    use_canonicalized_obj_meshes: bool = True
    # Downsample to 1fps
    downsample: bool = True
    # Preprocess only selected subjects
    subjects: List = field(default_factory=lambda: [])
    # Preprocess only selected objects
    objects: List = field(default_factory=lambda: [])
    # Threshold for computing human-object contacts (if no precomputed contacts are provided)
    contact_threshold: float = 0.050
    # Custom object names
    object_names: Dict[str, str] = field(default_factory=lambda: {
        "obj_01": "suitcase", "obj_02": "skateboard", "obj_03": "ball",
        "obj_04": "umbrella", "obj_05": "tennisracket", "obj_06": "briefcase",
        "obj_07": "chair", "obj_08": "bottle", "obj_09": "cup", "obj_10": "ottoman"
    })
    # Generate object keypoints after preprocessing
    generate_obj_keypoints: bool = False


@dataclass
class PreprocessOmomoConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "omomo/")
    target: str = os.path.join("${env.datasets_folder}", "omomo_smplh/")
    objects_path: str = os.path.join("${omomo.root}", "tools/canonicalized_decimated_objects")

    # Split (needed to use pre-defined splits)
    split: str = "train"  # "train", "test"
    # Use manualy canonicalized meshes (False is not supported)
    use_canonicalized_obj_meshes: bool = True
    # Downsample to 1fps
    downsample: bool = True
    # Preprocess only selected subjects
    subjects: List = field(default_factory=lambda: [])
    # Preprocess only selected objects
    objects: List = field(default_factory=lambda: [])
    # Threshold for computing human-object contacts (if no precomputed contacts are provided)
    contact_threshold: float = 0.050
    # Generate object keypoints after preprocessing
    generate_obj_keypoints: bool = False
# ==============================================================================

@dataclass
class PreprocessConfig:
    # environment
    env: EnvironmentConfig = EnvironmentConfig()

    # Type of data to process
    input_type: str = "smplh"  # smplh, hands, smpl
    # Number of points to sample from object mesh
    obj_keypoints_npoints: int = 1500
    # Align SMPL using skeleton information
    align_with_joints: bool = False
    # Scale to [-1, 1]
    normalize: bool = False
    # Align with ground plane
    align_with_ground: bool = False

    # datasets
    behave: PreprocessBehaveConfig = PreprocessBehaveConfig()
    grab: PreprocessGrabConfig = PreprocessGrabConfig()
    intercap: PreprocessIntercapConfig = PreprocessIntercapConfig()
    omomo: PreprocessOmomoConfig = PreprocessOmomoConfig()
