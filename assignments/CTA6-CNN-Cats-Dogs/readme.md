# CTA6 – Convolutional Neural Network: Cats vs. Dogs

This project was completed for **CSC580 – Deep Learning** as part of my **Master of Science in Artificial Intelligence at Colorado State University Global**.

The assignment explores the use of a **Convolutional Neural Network (CNN)** for image classification. The objective is to train a neural network to distinguish between images of cats and dogs while examining how convolutional architectures learn spatial features directly from image data.

This project extends the course's progression from traditional machine learning and fully connected neural networks into **computer vision and convolutional deep learning**.

---

## Project Objective

The primary objective is to develop and evaluate a CNN capable of performing binary image classification:

```text
Input Image
     │
     ▼
Convolutional Neural Network
     │
     ▼
┌─────────┐
│ Cat/Dog │
└─────────┘
```

Unlike the structured datasets used in previous CSC580 assignments, the input features in this project are the image pixels themselves. The CNN must learn useful visual representations during training rather than relying on manually defined features.

---

## Dataset

The assignment specifies the **Kaggle Dogs vs. Cats** competition dataset.

The original training dataset contains:

```text
25,000 labeled images

12,500 cats
12,500 dogs
```

Images vary in resolution, orientation, background, lighting, animal position, and other visual characteristics, creating a realistic binary image-classification problem.

### Dataset Access Note

During implementation, the Kaggle competition interface repeatedly redirected dataset download requests to the competition rules page despite acceptance of the required competition terms.

Programmatic access was also attempted using an authenticated KaggleHub client. Authentication was successfully established, but the legacy competition endpoint returned an HTTP **403 Forbidden** response.

To preserve the intended classification task while maintaining reproducibility, this project uses TensorFlow's filtered distribution of the Kaggle Dogs vs. Cats training images.

The filtered distribution contains approximately **23,000 images**, with a small number of corrupted images from the original dataset removed.

This change does not alter the underlying cat-versus-dog classification problem and is documented to ensure transparency and reproducibility.

---

## Why Use a CNN?

Convolutional neural networks are particularly well suited to image data because they preserve and learn from the spatial relationships between neighboring pixels.

Rather than manually defining characteristics such as:

```text
Ear shape
Fur texture
Eye position
Face structure
Body shape
```

a CNN learns hierarchical visual representations automatically.

Conceptually:

```text
Raw Pixels
    │
    ▼
Edges / Lines
    │
    ▼
Textures / Shapes
    │
    ▼
Animal Features
    │
    ▼
Higher-Level Representation
    │
    ▼
Cat or Dog
```

Early convolutional layers can learn relatively simple patterns such as edges and textures, while deeper layers can combine these representations into increasingly complex visual features useful for classification.

---

## Planned Workflow

The project will include:

* Acquisition and preparation of the Dogs vs. Cats dataset
* Inspection of image dimensions and class distribution
* Image resizing and preprocessing
* Separation of training and validation data
* Construction of a convolutional neural network
* Convolution and pooling layers
* Binary classification
* Model training
* Validation-performance monitoring
* Training and validation accuracy visualization
* Training and validation loss visualization
* Evaluation of model generalization
* Analysis of overfitting
* Prediction of previously unseen images

---

## CNN Architecture

The final architecture will be documented after implementation.

The general model structure will follow the pattern:

```text
Input Image
     │
     ▼
Convolution
     │
     ▼
Activation
     │
     ▼
Pooling
     │
     ▼
Convolution
     │
     ▼
Activation
     │
     ▼
Pooling
     │
     ▼
Feature Representation
     │
     ▼
Dense Classification Layer
     │
     ▼
Cat / Dog Probability
```

Specific layer sizes, image dimensions, activation functions, optimization parameters, and training configuration will be documented after experimentation.

---

## Model Evaluation

Model performance will be evaluated using separate training and validation data.

Particular attention will be given to the relationship between:

```text
Training Accuracy
        vs.
Validation Accuracy
```

and:

```text
Training Loss
        vs.
Validation Loss
```

A substantial divergence between training and validation performance may indicate **overfitting**, where the CNN learns the training images extremely well but fails to generalize effectively to previously unseen images.

---

## Connection to Previous Coursework

CTA6 follows several experiments examining the relationship between model architecture, optimization, and generalization.

Earlier CSC580 assignments explored:

* Linear regression
* Fully connected neural networks
* Regression with TensorFlow
* Biomedical toxicity classification
* Random Forest classification
* Neural network hyperparameter optimization
* Early stopping
* Overfitting
* Model generalization

CTA5 provided an especially important comparison between traditional machine learning and deep learning. Despite extensive neural network optimization, a Random Forest classifier generalized better on the structured Tox21 molecular dataset.

CTA6 introduces a different type of problem.

Rather than using structured molecular features, the model must learn useful representations directly from **raw image data**. CNNs are specifically designed to exploit this spatial structure, making this assignment an opportunity to examine a problem for which deep learning is particularly well suited.

---

## Broader Research Relevance

Although this assignment uses cats and dogs, CNNs have substantial applications in healthcare.

Examples include:

* Radiology image analysis
* Pathology slide classification
* Dermatological image analysis
* Retinal imaging
* Medical image segmentation
* ECG and physiological signal analysis using 1D convolutions
* Pattern recognition within continuous wearable data

One area of particular interest is whether convolutional approaches could identify local temporal patterns within longitudinal wearable measurements such as heart rate, heart-rate variability, sleep, activity, temperature, and respiratory data.

However, an important lesson from previous coursework is that **the most sophisticated model is not automatically the best model**. CNNs are compelling when the structure of the data makes convolution useful, but their performance should still be compared with other appropriate machine-learning approaches.

The objective of applied AI should therefore be to select the architecture that most reliably solves the underlying problem rather than using deep learning simply because it is available.

---

## Technologies

* Python 3.11
* TensorFlow 2.21
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* Kaggle / KaggleHub
* PyCharm
* Git
* GitHub

---

## Project Structure

```text
CTA6-CNN-Cats-Dogs/
├── cnn_cats_dogs.py
├── README.md
├── screenshots/
└── documentation/
```

The image dataset is intentionally excluded from version control because of its size and external distribution requirements.

---

## Status

**Project Status:** In Progress

Current progress:

* Project repository structure created
* Kaggle dataset access investigated
* Kaggle API authentication configured
* Legacy competition access limitation identified
* Alternative TensorFlow-hosted dataset selected
* CNN implementation pending
* Model training and evaluation pending
* Final analysis pending

This README will be updated as the model is developed and evaluated.

---

## Author

**Christine DeLuna**

Master of Science in Artificial Intelligence  
Colorado State University Global

Founder & CEO, AuVentures Health