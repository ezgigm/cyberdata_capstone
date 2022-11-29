# Facial Detection and Recognition for Security Applications

## Problem

Facial recognition is challenging and yet a very useful technology if applied well. 

## Aim

In this project, we aimed to make a robust facial recognition system using state of the art models and methods to achieve high performance and solve the problem of spoofing. 

## Solution

ArcFace - anti-spoofing techniques

## Data

The dataset contains headshot photos of people in the plain background from Adobe stock, and can be found in this Drive folder.(add a link)

## Methodology

1. Preprocessing
    *Facial detection and cropping
    *Facial alignment 
    *Uniform resizing 
 
Verification
 Anti-spoofing
At this stage, we will detect texture as either image or printed image. This helps us to differentiate the real person and the printed photo of the person. 
In order to create a reasonable larger dataset for our model to learn, data augmentation was needed
For photo images
Horizontal and vertical flips
90 degree and -90 degree rotations
 For print images
Auto enhance
Horizontal and vertical flips 
90 degree and -90 degree rotations
LBP algorithm will be used to get feature vectors distinct to each texture
SVM or other models will learn features in order to classify whether this is an image or a printed paper
Hyperparameter tuning
 
Matching
Arcface
State of the art method that uses a special loss function called Additive Angular Margin penalty, it will do the facial recognition
 ArcFace head with ResNet backbone, and a function that detect the distance between two images so that it can classify if the person is registered or not.


## Findings 

## Future Improvements

# Repository Guide




