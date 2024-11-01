# Deep Learning Demos

This repository contains various deep learning demos, including a simple CNN and a YOLO example for object detection.

## Table of Contents

-   [Project Description](#project-description)
-   [Installation](#installation)
-   [Usage](#usage)
    -   [Simple CNN](#simple-cnn)
    -   [YOLO Example](#yolo-example)

## Project Description

This project aims to provide hands-on examples of deep learning applications. The demos included are:

-   **Simple CNN**: A basic Convolutional Neural Network for image classification tasks. This demo helps users understand the fundamental concepts of CNNs and how they can be applied to classify images.
-   **YOLO Example**: A real-time object detection model that uses the YOLO algorithm. This demo illustrates how to easily detect multiple objects in images or video streams by leveraging a pre-trained model.

## Installation

To run the demos, you need to install the required dependencies. You can either install the requirements using the `requirements.txt` file or using the following command to install them:

```bash
pip install torch torchvision tqdm opencv-python matplotlib
```

## Usage

### Simple CNN

The Simple CNN demo demonstrates how to build and train a basic convolutional neural network for image classification.

```bash
# Command to run the Simple CNN demo
python simple_cnn_demo.py
```

### YOLO Example

The YOLO Example demo shows how to use the YOLO model for object detection using your camera feed.

```bash
# Command to run the YOLO Example demo
python yolo_example.py
```

#### COCO 2017 Dataset

YOLOv5 uses the COCO 2017 dataset. You can check out the COCO dataset and the classes it uses here:

-   [COCO 2017](https://www.kaggle.com/datasets/ultralytics/coco128)
-   [Classnames](https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda)
