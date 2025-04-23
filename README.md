# Neural Network Training with PyTorch 

This project implements and trains several neural network architectures using **PyTorch** as part of the CS189 machine learning course at UC Berkeley.

## Overview

The main focus of this assignment was to gain hands-on experience with PyTorch and understand the practical aspects of training neural networks. This includes:

- Designing and training multi-layer perceptrons (MLPs)
- Implementing feedforward and backpropagation logic
- Applying techniques like ReLU, softmax, dropout, and weight initialization
- Monitoring training/testing accuracy and loss

All work is completed in a Colab notebook with free GPU acceleration.

## Files

- `Copy_of_CS189_HW_NN.ipynb`: Main Colab notebook containing all training and evaluation code
- `datasets/`: Folder containing required training and testing datasets (not included here due to size)

## Getting Started

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Make sure to enable GPU via:
   `Runtime > Change runtime type > Hardware accelerator > GPU`
3. Upload the `datasets/` folder or mount from Google Drive
4. Run each cell to train the models and analyze performance

## Key Concepts

- PyTorch Tensors and Autograd
- Training MLPs using SGD and Adam optimizers
- Loss functions: CrossEntropyLoss, MSELoss
- Epoch-based training, model evaluation, and hyperparameter tuning

## Requirements

- Python 3.x
- PyTorch
- Google Colab (recommended)

## Notes

- Training performance may vary depending on GPU availability and runtime environment.
- You are welcome to run this notebook locally if you have PyTorch and CUDA setup.

## Acknowledgements

This assignment is part of the CS189/289A: Introduction to Machine Learning course at UC Berkeley.
