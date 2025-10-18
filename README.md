<!-- HEADER -->
<p align="center">
    <h1 align="center">TriDi: Trilateral Diffusion of 3D Humans, Objects and Interactions</h1>
    <!-- authors -->
    <p align="center">
        <a href="https://virtualhumans.mpi-inf.mpg.de/people/Petrov.html"><b>Ilya A. Petrov</b></a>
        &emsp;
        <a href="https://riccardomarin.github.io/"><b>Riccardo Marin</b></a>
        &emsp;
        <a href="https://virtualhumans.mpi-inf.mpg.de/people/Chibane.html"><b>Julian Chibane</b></a>
        &emsp;
        <a href="https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html"><b>Gerard Pons-Moll</b></a>
    </p>
    <!-- conference -->
    <h3 align="center">ICCV 2025</h3>
    <!-- teaser -->
    <p align="center">
        <img src="assets/petrov25tridi.gif" alt="Project Teaser" width="600px">
    </p>
    <!-- badges -->
    <p align="center">
        <a href="https://arxiv.org/abs/2412.06334">
            <img src="https://img.shields.io/badge/arXiv-2412.06334-b31b1b.svg?style=for-the-badge" alt="Paper PDF">
        </a>
        &emsp;
        <a href="https://virtualhumans.mpi-inf.mpg.de/tridi/">
            <img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=Google%20chrome&logoColor=white" alt="Project Page">
        </a>
        &emsp;
        <a href="https://youtu.be/_UHLfbgGCbI">
            <img src="https://img.shields.io/badge/YouTube-video-black?style=for-the-badge&logo=youtube&logoColor=white&labelColor=FF0000&color=black" alt="YouTube video">
        </a>
    </p>
</p>


## Environment
The code was tested under `Ubuntu 24.04, Python 3.10, CUDA 13.0, PyTorch 2.9.0`.
Use the following command to create a conda environment with necessary dependencies:
```bash
conda env create -f environment.yml
```

## Data downloading and processing
The steps are described in [docs/data.md](./docs/data.md).


## Pre-trained models and evaluation
Pre-trained model can be obtained from the [link](https://nc.mlcloud.uni-tuebingen.de/index.php/s/bmsRACRqzCQ4QPq). 
With the commands:
```bash
wget https://nc.mlcloud.uni-tuebingen.de/public.php/dav/files/bmsRACRqzCQ4QPq/gb_main.pth -O ./assets/gb_main.pth
wget https://nc.mlcloud.uni-tuebingen.de/public.php/dav/files/bmsRACRqzCQ4QPq/gb_contacts.pth -O ./assets/gb_contacts.pth
```

The command below is used to run sampling. Prameter `sample.mode` controls the choice of modalities, i.e.:
three numbers correspond to human, object, and interaction, respectively; 
`1` means the modality is sampled, `0` means it is conditioned on. 
For example, `sample.mode="sample_101"` means sampling human and interaction conditioned on the object.
```bash
python main.py -c config/env.yaml scenarios/gb_main.yaml -- \
  run.job=sample run.name=001_gb_main sample.target=hdf5 \
  resume.checkpoint="./assets/gb_main.pth" \
  dataloader.batch_size=1024 sample.mode="sample_101" \
  run.datasets=["grab","behave"] sample.dataset=normal sample.repetitions=3 \
  model.cg_apply=True model.cg_scale=2.0
```
Use the command below to run evaluation on the generated samples. The `eval.sampling_target` parameter controls 
which modalities are evaluated (possible values: `sbj_contact`, `obj_contact`,):
```bash
python main.py -c config/env.yaml scenarios/gb_main.yaml -- \
  run.job=eval run.name=001_gb_main resume.step=-1 \
  eval.sampling_target=['sbj_contact'] 
```

## Training
Use the following command to run the training:
```bash
python main.py -c config/env.yaml scenarios/gb_main.yaml -- \
  run.name=001_gb_main run.job=train
```


## Citation
```bibtex
@inproceedings{petrov2025tridi,
   title={TriDi: Trilateral Diffusion of 3D Humans, Objects and Interactions},
   author={Petrov, Ilya A and Marin, Riccardo and Chibane, Julian and Pons-Moll, Gerard},
   booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
   year={2025}
}
```


## Acknowledgements
This project benefited from the following resources:
* [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/license.html), [GRAB](https://grab.is.tue.mpg.de/), 
[OMOMO](https://github.com/lijiaman/omomo_release), and [InterCap](https://intercap.is.tue.mpg.de/) datasets;
* [grab](https://github.com/otaheri/GRAB) preprocessing code; 
* [smplx](https://github.com/vchoutas/smplx) repository: SMPL-X to SMPL+H conversion;
* [PC^2 diffusion](https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion): diffusion implementation;
* [PointNeXt](https://github.com/guochengqian/PointNeXt): point cloud encoder;
* [blendify](https://github.com/ptrvilya/blendify/): all visualizations;
* [blogpost](http://danshiebler.com/2016-09-14-parallel-progress-bar/): parallel map implementation.