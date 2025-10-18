## SMPL, SMPL+H and MANO model
The project uses [smplx](https://github.com/vchoutas/smplx) package to work with SMPL body model.
Follow [this](https://github.com/vchoutas/smplx#downloading-the-model) link for the instruction on setting it up.
After installing smplx make sure to set `smpl_folder=<path to folder with SMPL, SMPL+H, and MANO models>` in 
`config/env.yaml` and `config/preprocessing.toml` to point to the folder with the 
[following](https://github.com/vchoutas/smplx#model-loading) structure.


## Downloading data
### GRAB
Instructions on downloading and setting up the GRAB dataset.

1. Register and download data from the official [website](https://grab.is.tue.mpg.de/index.html). 
   Required data: Subject 1 - Subject 10 (GRAB parameters), Subject Shape Templates, and GRAB objects.
2. Extract data using official [extracting code](https://github.com/otaheri/GRAB/#getting-started), 
   the expected folder structure is provided below (`data_smplh` is generated during the next step). 
3. Convert SMPL-X parameters to SMPL+H (instructions are [here](../popup/utils/smplx2smplh/README.md)).

#### Folders structure
```bash
<extracted GRAB path>
├── grab
│   ├── s1
│   │   ├── airplane_fly_1.npz
│   │   ├── ...
│   ├── ...
│   ├── s10
├── tools
│   ├── object_meshes
│   │   ├── contact_meshes
│   │   │   ├── airplane.ply
│   │   │   ├── ...
│   ├── subject_meshes
│   ├── ...
├── data_smplh
│   ├── s1
│   │   ├── banana_eat_1
│   │   │   ├── sequence_data.pkl
│   │   ├── ...
│   ├── ...
│   ├── s10
```

### BEHAVE
Instructions on downloading and setting up the BEHAVE dataset.

1. Download data from the official [website](https://virtualhumans.mpi-inf.mpg.de/behave/license.html). 
   Required data: Scanned objects, and Annotations at 30fps.
2. Extract data, following the folder structure below.

#### Folders structure
```bash
<extracted BEHAVE path>
├── behave-30fps-params-v1
│   ├── sequences
│   │   ├── Date01_Sub01_backpack_back
│   │   ├── ...
├── objects
│   ├── backpack
│   ├── ...
```

### OMOMO
TBD

### InterCap
TBD

## Preprocessing data
Preprocessing scripts are located in `tridi/preprocessing/`. Before running the preprocessing scripts make sure to
update the environment configuration in `config/env.yaml`:
```
  datasets_folder: ./data/preprocessed/  # Folder to store preprocessed datasets
  raw_datasets_folder: ./data/raw/       # Folder with downloaded datasets (with subfolders for each dataset)
  assets_folder: ./assets/               # Folder with assets (usually ./assets/ within repo)
  experiments_folder: ./experiments      # Folder with experiments (usually ./experiments/ within repo)
  smpl_folder: ./data/smplx_models/      # Folder with smpl models
```

After that run the preprocessing scripts for each dataset. Key step is to generate object keypoints for each dataset 
that align with trained ones (done whe `-g` is passed to the script):
```bash
# GRAB
python -m hoigen.preprocessing.preprocess_grab -g -c ./config/env.yaml \
  grab.subjects=["s1","s2","s3","s4","s5","s6","s7","s8"] \
  grab.downsample="10fps" grab.split="train"
python -m hoigen.preprocessing.preprocess_grab -c ./config/env.yaml \
  grab.subjects=["s9","s10"] grab.downsample="1fps" grab.split="test"

# BEHAVE 30 fps
python -m hoigen.preprocessing.preprocess_behave_30fps -g -c ./config/env.yaml \
  behave.split="train" behave.downsample="10fps"
python -m hoigen.preprocessing.preprocess_behave_30fps -c ./config/env.yaml \
  behave.split="test" behave.downsample="1fps"

# BEHAVE 1 fps
python -m hoigen.preprocessing.preprocess_behave -c ./config/env.yaml \
  behave.split="train" behave.split_file="./assets/behave_only_1fps.json" \
  behave.combine_1fps_and_30fps=True behave.downsample="10fps"
python -m hoigen.preprocessing.preprocess_behave -c ./config/env.yaml \
  behave.split="test" behave.split_file="./assets/behave_only_1fps.json" \
  behave.combine_1fps_and_30fps=True behave.downsample="1fps"
  
# OMOMO
# TBD

# InterCap
# TBD
```