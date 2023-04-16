# Brain-Tumor-Segmentation using Unet and Flask

This project implements a brain tumor segmentation algorithm using the UNet architecture and provides a simple web interface using Flask. 
The goal is to allow users to upload a brain MRI images and get a segmented image with the tumor area highlighted.

Get the dataset [BraTS2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation). All BraTS multimodal scans 
are available as NIfTI files (.nii.gz) and describe a) native (T1) and
b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes.

The unet model takes 2 files:

1. FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.
2. T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.

concatenates them along depth, then fed into model for training using data generator with batch size = 1. 
The model is capable of identifing 3 major areas of Tumor:

1. NECROTIC/CORE
2. EDEMA
3. ENHANCING
4. (or) all the above.

Metrics on test dataset: loss: 0.0357 - accuracy: 0.9887 - mean_io_u: 0.3763 - dice_coef: 0.4627 - precision: 0.9927 - 
sensitivity: 0.9841 - specificity: 0.9975 - dice_coef_necrotic: 0.3058 - dice_coef_edema: 0.5723 - dice_coef_enhancing: 0.4697

## TO DO:
1. add css to make it interactive.
2. analyse the dataset on Unet++, U-net architecture with ResNet backbone, etc.

## Referneces:
1. Andrew Ng's Convolutional Neural Networks course on Coursera.
2. [Ronneberger et al. in U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597v1)
3. [Brain Tumor Segmentation using Enhanced U-Net Model with Empirical Analysis](https://arxiv.org/abs/2210.13336)
4. [kaggle](https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net)

