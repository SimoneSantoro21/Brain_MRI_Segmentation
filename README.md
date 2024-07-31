# Brain_MRI_Segmentation
This repository contains code and documentation relative to the Software and Computing & Pattern Recognition exams of the Applied Physics curriculum of Physics Master degree at UniBo.

- [Brain MRI Segmentation](#Brain_MRI_Segmentation)
  - [Overview](#overview)
  - [Contents](#contents)
  - [Pre-requisites](#pre-requisites)
  - [Tutorial](#tutorial)
    - [Installing](#installing)
    - [Testing](#testing)
    - [Training](#training)
    - [Prediction](#prediction)
    - [Evaluation](#evaluation)
  - [References](#references)

## Overview
Multiple Sclerosis (MS) is a chronic autoimmune disease that affects the central nervous system, leading to demyelination and subsequent neurodegeneration. It manifests through a wide range of neurological symptoms, including visual disturbances, muscle weakness, coordination problems, and cognitive deficits. The progression and severity of MS can vary significantly among individuals, making early and accurate diagnosis essential for effective treatment.
Fluid-attenuated inversion recovery (FLAIR) MRI can be used in the detection of MS lesions.

Accurate segmentation of MS lesions is essential for both diagnosis and research purposes. In diagnosis and monitoring, precise identification and measurement of lesions aid in diagnosing and tracking its progression. In research, high-quality segmentation is crucial for studies aiming to understand the pathophysiology of MS and develop new therapeutic approaches.

This project provides a U-Net model[1] for segmenting such lesions in brain FLAIR MRI scans, converted to a Bayesian approximation[2] by exploiting dropout layers. This approach, allowed to not only obtain the model prediction, but to be able to associate it with an uncertainty measure as well.

**Example of segmentation**. **Left:** Original image with ground truth. **Center:** Original image with predicted lesions. **Right** Original image with uncertainty.

<a href="https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/images/example_result.png">
  <div class="image">
    <img src="images\example_result.png" width="100%">
  </div>
</a>

## Contents
Brain MRI Segmentation is composed of scripts and modules:
- Scripts allows to prepare the dataset, train the model, inference predictions and evaluate the performances.
- Modules contain definitions of the pre-processing and evaluation metrics functions, as well as the classes needed to build the model and loss function in PyTorch.

Script description:

| **Script** | **Description** |
|:----------:|:---------------:|
| [prepare_dataset](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/prepare_dataset.py) | Organizes patient data by extracting axial slices and saving them in training, validation and testing directories. This sript is ONLY intended to be used to prepare tutorial dataset|
| [training](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/training.py) | Performs model training with specified data directory, model save path, learning rate, epochs and batch size|
| [predict](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/predict.py) |Performs inference on an input image, showing the output image and variance along the original image with ground truth and saving the results in a "predictions" directory|
| [evaluate](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/evaluate.py) |Evaluates prediction quality by computing metrics implemented in evaluation_metrics.py|


Modules description:

| **Module**| **Description**|
|:---------:|:--------------:|
| [pre_processing_functions](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/libs/pre_processing_functions.py) | Contains functions for pre-processing images: grey level normalization, resizing and adjusting spacing|
| [evaluation_metrics](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/libs/evaluation_metrics.py) | Contains functions for evaluation metrics: precision, accuracy, recall, jaccard and dice|
| [dataset](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/libs/dataset.py) | Torch Dataset class for loading dataset to the model |
| [U_Net_components](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/libs/U_Net_components.py) |Contains model components: DoubleConvolution operation, DownSampling block, UpSampling block| 
| [U_Net](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/libs/U_Net.py) |Contais the actual model, built by putting toghether the single components| 
| [focal](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/libs/focal.py) |Class for the PyTorch implementation of the focal loss function[3]

## Pre-requisites
To be able to make the model work is mandatory to install PyTorch. For installing PyTorch, please refere to the original documentation at:

https://pytorch.org/get-started/locally/.

For what concernes other libraries, they can be inspected in [requirements.txt](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/blob/main/requirements.txt). It is convenient to create a Conda environment in which to install PyTorch and the other libraries, that can be installed by activating the environment and running

```bash
conda install requirements.txt
```

## Tutorial


### Installing
For installing the software, the user needs to clone this repository:

```bash
git clone https://github.com/SimoneSantoro21/Brain_MRI_Segmentation
```

### Testing

Testing routines use ```PyTest``` that needs to be installed
 to perform the test.

A full set of test is provided in [testing](https://github.com/SimoneSantoro21/Brain_MRI_Segmentation/tree/main/test) directory.
It is possibe to execute all the tests by using:

```bash
python -m pytest
```
### Organizing data
The model expects the input dataset directory to be organized as follows: 
- dataset
    - Training
    - Validation
    - Testing

Each subdirectory (Training, Validation, Testing) must contain two further subdirectories named 'FLAIR' and 'LESION', each holding 2D axial brain slices in the respective image modalities.

This tutorial uses the open-access Mendeley dataset[4] comprising Multiple Sclerosis (MS) MRI brain scans, available at the following link:

https://data.mendeley.com/datasets/8bctsm8jz7/1

The dataset is structured into 60 patient-specific directories, each containing:

- Flair.nii: A 3D NIfTI-formatted MRI scan in FLAIR modality.
- LesionSeg-Flair.nii: A corresponding 3D NIfTI image with lesion segmentation masks derived from the FLAIR scan.

After downloading and extracting the dataset into the project's directory, execute the prepare_dataset script by using:

```bash
python -m prepare_dataset
```
This script will organize the data into a "dataset" directory suitable for the subsequent tutorial steps.

### Training
Once the dataset is well organized, it is possible to train the model by using the training script. Is is required to specify the dataset path and the model save path with the following command:

```bash
python -m training --input "dataset" --output "model/unet.pth"
```

The following training parameters can also be specified:
- Learning rate (--lr): default value 5e^-3
- epochs (--epochs): default value 100
- batch size (--batch): default value 8

### Prediction
For prediction, the user trained model as well as the pre-trained one can be used by running predict.py. Let's say that we are interested in predicting lesions in the image FLAIR_5 of the testing dataset, then the command should be the following:

```bash
python -m predict --input "dataset/testing" --index 5 --model "model/unet.pth"
```
The following parameters can also be specified:
- Threshold (--threshold): default value 0.5
- Plot (--plot): default value True
- Forward (--forward): default value 50

After running the command, a directory named "predictions" will be created, with a sub-directory named "Prediction_index-5" containing the following images in .png format:
- comparison: Image showing the comparison between ground truth and prediction, along variance
- mean_prediction: Binary prediction output
- variance_prediction: Variance output in 'hot' color map.

### Evaluation
After the prediction, it is now possible to evaluate the model performance by running: 

```bash
python -m evaluate --index 5
```

The output will be similar to

| Metric            | Value      |
|-------------------|-------------|
| Precision score   | 0.721 |
| Recall score      | 0.660    |
| Accuracy          | 0.997     |
| Jaccard Index     | 0.526  |
| Dice coefficient  | 0.689   |


## References
<blockquote>[1] -Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. 2015. doi: 10.1007/978-3-319-24574-4\_28.
</blockquote>

<blockquote>[2] - Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. Ed. by Maria Florina Balcan and Kilian Q. Weinberger. 2016. url: https://proceedings.mlr.press/v48/gal16.html.
</blockquote>

<blockquote>[3] - Tsung-Yi Lin et al. Focal Loss for Dense Object Detection. 2018. arXiv: 1708.02002 [cs.CV].21
</blockquote>

<blockquote>[4] - M Muslim, Ali (2022), “Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information”, Mendeley Data, V1, doi: 10.17632/8bctsm8jz7.1 
</blockquote>


