from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    datasets_folder: str = "./data/preprocessed/"
    raw_datasets_folder: str = "./data/raw/"
    assets_folder: str = "./assets/"

    experiments_folder: str = './experiments'
    smpl_folder: str = "./data/smplx_models/"