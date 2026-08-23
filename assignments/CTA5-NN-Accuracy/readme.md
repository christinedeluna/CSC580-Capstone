# CSC580 – Improving Neural Network Accuracy

## Tox21 Hyperparameter Optimization and Model Comparison

This project was completed for **CSC580 – Deep Learning** as part of my **Master of Science in Artificial Intelligence at Colorado State University Global**.

The assignment builds upon the Tox21 toxicity-classification neural network developed earlier in the course. The objective was to investigate whether systematic hyperparameter optimization could improve neural network performance and to compare the optimized model with a traditional Random Forest classifier.

An important outcome of the experiment was that although hyperparameter tuning improved the neural network, the Random Forest ultimately demonstrated stronger generalization to unseen data.

---

## Project Objectives

The project evaluates how neural network architecture and training hyperparameters influence classification performance on the **DeepChem Tox21 dataset**.

The experiment included:

- Loading and preprocessing the Tox21 molecular dataset
- Establishing a Random Forest baseline
- Building a configurable fully connected neural network
- Modernizing legacy TensorFlow 1.x instructional code for TensorFlow 2.21
- Evaluating multiple neural network architectures
- Tuning learning rate and dropout
- Evaluating training epochs and batch size
- Testing positive-class weighting
- Repeating configurations across multiple random seeds
- Comparing validation and test-set performance
- Evaluating model generalization and overfitting

---

## Dataset

The project uses the **Tox21 dataset provided through DeepChem**.

Each molecular compound is represented using:

- **1,024 molecular fingerprint features**
- Binary toxicity labels
- Sample weights
- Predefined training, validation, and testing datasets

The first Tox21 prediction task was used for this experiment.

Dataset dimensions:

```text
Training:     6,258 × 1,024
Validation:     782 × 1,024
Testing:        783 × 1,024