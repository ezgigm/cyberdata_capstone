# Facial Detection and Recognition for Security Applications
# NOTE: This Repo and Readme is IN PROGRESS...
## Problem

Facial recognition is challenging and yet a very useful technology if applied well. __More information is coming....__

## Aim

In this project, we aimed to make a robust facial recognition system using state of the art models and methods to achieve high performance and solve the problem of spoofing. 

## Solution

ArcFace - anti-spoofing techniques

## Data

The dataset contains headshot photos of people in the plain background from Adobe stock, and can be found in this Drive folder.(https://drive.google.com/drive/folders/1EYGouovWZR1JJCD4Yts7MnSMMDO3ODu4?usp=share_link)
The test dataset contains headshot, half and full body shot photos of Chinese people in the various background from V3 dataset, and can be found in this Drive folder. (https://drive.google.com/drive/folders/1Y8Ykn1fGMD9NmBSSyFeTHqSvKEb7Ru0h?usp=share_link)

## Checkpoints

The checkpoint file can be found in this Drive folder. (https://drive.google.com/drive/folders/1j0rRVoSuOvnCuP6bIaXP1XNnaaq_2sf4?usp=share_link)

## Methodology

1. Preprocessing
    * Facial detection and cropping
    * Facial alignment 
    * Uniform resizing 
 
2. Verification : Anti-spoofing
    * At this stage, we will detect texture as either image or printed image. This helps us to differentiate the real person and the printed photo of the person. 
    * In order to create a reasonable larger dataset for our model to learn, data augmentation was needed
      * For photo images
         * Horizontal and vertical flips
         * 90 degree and -90 degree rotations
      * For print images
         * Auto enhance
         * Horizontal and vertical flips 
         * 90 degree and -90 degree rotations
   * LBP algorithm will be used to get feature vectors distinct to each texture
      * SVM or other models will learn features in order to classify whether this is an image or a printed paper
   * Hyperparameter tuning
 
3. Matching
   * Arcface: State of the art method that uses a special loss function called Additive Angular Margin penalty, it will do the facial recognition
 ArcFace head with ResNet backbone, and a function that detect the distance between two images so that it can classify if the person is registered or not.

## Findings 

Findings from the results will be added here. 

## Future Improvements

Next steps and the improvement points will be added to this session. 

# Repository Guide

This section will show the links of the related part of the project. 




