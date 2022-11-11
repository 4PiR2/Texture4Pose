# Texture4Pose -- Master's Thesis

6 DoF pose estimation from monocular RGB images is one of the fundamental tasks in computer vision. However, its accuracy is often limited due to the lack of textures on the objects. In this thesis, we present a novel method to optimize the textures of the objects for pose estimation tasks. Our method is based on deep learning, and we use an end-to-end pipeline to optimize the object’s texture and pose estimator simultaneously. The pipeline consists of a texture generation network based on the texture field representation, a differentiable renderer, and a pose estimation network. The pipeline is trained on purely synthetic data and is evaluated on both synthetic and real data. Our experiments show that using a deep neural network to automatically perform local optimization is feasible to generate a texture with good pose estimation accuracy without much human effort. And the network-generated texture outperforms our hand-designed baseline textures.

## Thesis PDF

TBD

## Presentation Slides

TBD

## Usage

For train / test: run script `main.py`

Pretrained weights: download and put them into `weights` directory

For fine-grained configurations: make changes in `config` directory

This project runs on Python 3.9.12 with the following packages:

| Package                | Version     |
|------------------------|-------------|
| detectron2             | 0.6         |
| matplotlib             | 3.5.1       |
| ExifRead               | 3.0.0       |
| numpy                  | 1.23.4      |
| opencv-contrib-python  | 4.5.5.64    |
| opencv-python          | 4.6.0.66    |
| opencv-python-headless | 4.5.5.64    |
| pandas                 | 1.4.1       |
| Pillow                 | 9.1.1       |
| pillow-heif            | 0.6.0       |
| Pillow-SIMD            | 9.0.0.post1 |
| pyheif                 | 0.7.0       |
| pyro-ppl               | 1.8.1       |
| pytorch-lightning      | 1.6.4       |
| pytorch3d              | 0.6.2       |
| sahi                   | 0.11.1      |
| scikit-image           | 0.18.3      |
| scikit-learn           | 1.0.2       |
| scipy                  | 1.7.3       |
| tensorboard            | 2.9.0       |
| tensorboardX           | 2.5         |
| torch                  | 1.11.0      |
| torchdata              | 0.3.0       |
| torchvision            | 0.12.0      |
| tqdm                   | 4.64.0      |
