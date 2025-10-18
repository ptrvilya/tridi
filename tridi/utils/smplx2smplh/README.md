# Conversion of GRAB data from SMPL-X to SMPL+H

## Table of Contents
  * [Description](#description)
  * [Requirements](#requirements)
  * [Using the code](#using-the-code)
    * [Data](#data)
    * [Conversion](#conversion)
  * [Citation](#citation)
  * [Licenses](#licenses)
    * [SMPL-X](#smpl-x)
    * [Conversion](#conversion)


## Description
The folder contains code for converting the GRAB dataset human body model parameters from SMPL-X to SMPL+H.

This code is a simplified implementation of 
[transfer model](https://github.com/vchoutas/smplx/tree/main/transfer_model) 
from the official [SMPL-X](https://github.com/vchoutas/smplx) repository with some functions taken from 
[GRAB preprocessing code](https://github.com/otaheri/GRAB/blob/master/grab/grab_preprocessing.py).


## Using the code

### Data
Register on the [SMPL-X website](http://smpl-x.is.tue.mpg.de/), go to the downloads section to get the correspondences
and sample data, by clicking on the *Model correspondences* button.

Extract the downloaded zip to `transfer_data` folder. You should have the following folder structure now:

```bash
transfer_data
├── meshes
│   ├── smpl
│   ├── smplx
├── smpl2smplh_def_transfer.pkl
├── smpl2smplx_deftrafo_setup.pkl
├── smplh2smpl_def_transfer.pkl
├── smplh2smplx_deftrafo_setup.pkl
├── smplx2smpl_deftrafo_setup.pkl
├── smplx2smplh_deftrafo_setup.pkl
├── smplx_mask_ids.npy
```

### Converion
To run the code to convert SMPL-X meshes to SMPL+H parameters use the following command:
```Shell
python -m main -g <extracted GRAB path> -s <path to folder with SMPL, SMPL+H, and MANO models> 
```

Note: `Lines 183-197` in `convert_grab_smplx2smplh.py` contain some pre-set config parameters (like `batch_size`), 
change them according to the available hardware.
This should be run from the top directory of the repository.


## Citation
Depending on which model is loaded for your project, i.e. SMPL-X or SMPL+H or SMPL, please cite the most relevant work:
```bibtex
  @article{SMPL:2015,
    author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
    title = {{SMPL}: A Skinned Multi-Person Linear Model},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    month = oct,
    number = {6},
    pages = {248:1--248:16},
    publisher = {ACM},
    volume = {34},
    year = {2015}
  }

  @article{MANO:SIGGRAPHASIA:2017,
    title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
    author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    volume = {36},
    number = {6},
    pages = {245:1--245:17},
    series = {245:1--245:17},
    publisher = {ACM},
    month = nov,
    year = {2017},
    url = {http://doi.acm.org/10.1145/3130800.3130883},
    month_numeric = {11}
  }

  @inproceedings{SMPL-X:2019,
    title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
    author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
  }
```

If you use the GRAB dataset, please cite the following work:
```bibtex
  @inproceedings{GRAB:2020,
    title = {{GRAB}: A Dataset of Whole-Body Human Grasping of Objects},
    author = {Taheri, Omid and Ghorbani, Nima and Black, Michael J. and Tzionas, Dimitrios},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2020},
    url = {https://grab.is.tue.mpg.de}
  }
```


## Licenses
### SMPL-X
Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [LICENSE file](https://github.com/vchoutas/smplx/blob/master/LICENSE) 
and any accompanying documentation before you download and/or use the SMPL-X/SMPLify-X model, data and software, 
(the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations.
By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this
github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be
bound by them. If you do not agree with these terms and conditions, you must not download and/or use 
the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights
under this [LICENSE](https://github.com/vchoutas/smplx/blob/master/LICENSE).

### GRAB
Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [LICENSE file](https://github.com/otaheri/GRAB/blob/master/LICENSE) for the terms and 
conditions and any accompanying documentation  before you download and/or use the GRAB data, model and software, 
(the "Data & Software"), including 3D meshes (body and objects), images, videos, textures, software, scripts, and
animations. By downloading and/or using the Data & Software (including downloading, cloning, installing, and any 
other use of the corresponding github repository), you acknowledge that you have read and agreed to the LICENSE 
terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and 
conditions, you must not download and/or use the Data & Software. Any infringement of the terms of
this agreement will automatically terminate your rights under this [LICENSE](https://github.com/otaheri/GRAB/blob/master/LICENSE).
