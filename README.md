# DogBreedClassifier
_Model used for the final project of TI2022 - Dog Breed classifier app for local Veterinary_ . 

A model for classifying 120 Dog Breeds built using the [EfficientNet-B3](https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1) feature vector pre-trained on Imagenet (ILSVRC-2012-CLS) and fine-tuned for the task at hand. The model achieves an overall accuracy of 86.7%

## Getting started 🚀

To get a working environment there are two possible options. 

1. Create a conda environment with the listed pre-requisites
2. Create a conda environment from the .yml file

### Pre-requisites 📋

_Software and dependencies needed_

```
pytorch-1.7.0
tqdm-4.54.1
torchvision-0.8.1
numpy-1.19.2
```

### Installation 🔧

To get started make sure you either have the listed pre-requisites or set up the anaconda environment from the .yml file.

```bash
conda env create -f environment.yml
```

Make sure you activate the environment. 
```bash
conda activate DogNet
```

And verify that it was properly installed.
```bash
conda env list
```

## Data

The dataset used to train the model was the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) which contains 20,580 images with a total of 120 dog breeds.


## Acknowledgments 🎁

Special thanks to Aladding Persson [aladdinpersson](https://github.com/aladdinpersson) for his tutorial on [Building a Dog Breed Identifier App from scratch - DogNet](https://youtu.be/XU5rTgfnq6E)
