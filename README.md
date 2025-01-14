# Description:
This project is all about fixing compatibility problems in an older TensorFlow/Keras repo to make it work with newer versions. I’ve updated the code, tested it, and set everything up in a Python virtual environment to make it easy to run. There’s also a Conda environment file (environment.yml) if you prefer using Conda to set things up on your own system.

## What’s in the project?

    Fixed code for modern TensorFlow/Keras versions.
    A virtual environment so you don’t have to worry about dependency headaches.
    A Conda environment file for those who want extra portability.

## Setup & getting started

1. **Conda Env Setup**
   ```bash
   conda env create -f environment.yml
   conda activate your-env-name
   ```
2. **Weights**
   Download the pre-trained weights inside the root directory (maskrcnn). The weights can be downloaded from this link: https://github.com/ahmedfgad/Mask-RCNN-TF2/releases/download/v3.0/mask_rcnn_coco.h5.

3. **getting started**
   - store images in the "test_image" or the "image_test" directories (a choice is given later which directory to run in)
   - if not already in the root directory ```cd maskrcnn```
   - run ```python maskrcnn_predict.py```
