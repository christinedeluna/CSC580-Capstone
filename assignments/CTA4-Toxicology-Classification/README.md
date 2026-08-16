# CSC580 – Deep Learning Capstone

This repository contains coursework, experiments, and programming assignments completed for **CSC580 – Deep Learning** as part of my **Master of Science in Artificial Intelligence** at **Colorado State University Global**.

The course explores modern deep learning techniques using TensorFlow and Keras, including neural network design, supervised learning, optimization, model evaluation, and predictive analytics. Throughout the course, I focus on understanding not only how to implement neural networks, but also why they work and how they can be applied to real-world biomedical problems.

---

## Repository Structure

```text
CSC580/
│
├── assignments/
│   ├── CTA1-LinearRegression/
│   ├── CTA2-NeuralNetwork/
│   ├── CTA3-Regression/
│   └── CTA4-Toxicology-Classification/
│
├── experiments/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Completed Assignments

### CTA1 – Linear Regression

Developed a supervised learning model using Scikit-learn's `LinearRegression` to learn an algebraic function from generated training data.

**Concepts**

- Feature engineering
- Regression
- Model fitting
- Prediction

---

### CTA2 – Feedforward Neural Networks

Built a fully connected neural network using TensorFlow/Keras to predict future video game revenue from historical sales data.

**Concepts**

- Data normalization
- Dense neural networks
- Model training
- Model persistence
- Regression using neural networks

---

### CTA3 – Deep Learning Regression

Expanded the neural network implementation through additional regression experiments, hyperparameter tuning, and evaluation techniques while reinforcing TensorFlow workflows.

**Concepts**

- Neural network optimization
- Hyperparameter tuning
- Model evaluation
- TensorFlow experimentation

---

### CTA4 – Toxicology Classification

Designed and implemented a feedforward neural network to classify chemical compounds as toxic or non-toxic using the **DeepChem Tox21** benchmark dataset.

This project required modernizing legacy TensorFlow 1.x instructional code for compatibility with **TensorFlow 2.21**, resolving DeepChem and RDKit compatibility issues, implementing mini-batch gradient descent, monitoring model convergence with TensorBoard, and evaluating model performance on unseen validation data.

**Highlights**

- DeepChem Tox21 dataset
- Molecular fingerprint representation (1,024 features)
- Binary toxicity classification
- Hidden layer with ReLU activation
- Dropout regularization
- Adam optimization
- Binary cross-entropy loss
- Mini-batch gradient descent
- TensorBoard visualization
- Validation accuracy: **90.92%**

---

## Technologies

- Python 3.11
- TensorFlow 2.21
- Keras
- DeepChem
- RDKit
- Scikit-learn
- NumPy
- Pandas
- TensorBoard
- Matplotlib

---

## Skills Demonstrated

- Neural network design
- Deep learning workflows
- TensorFlow graph construction
- Binary classification
- Mini-batch gradient descent
- Dropout regularization
- Model convergence analysis
- TensorBoard visualization
- Scientific Python dependency management
- Model evaluation and validation

---

## Research Interests

Although these assignments use educational datasets, the concepts directly support my long-term research interests in artificial intelligence for healthcare. My work focuses on developing AI systems capable of identifying longitudinal health patterns, assisting clinical reasoning, and supporting earlier recognition of complex medical conditions through ethical and explainable machine learning.

---

## Running the Projects

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Each assignment contains its own README with project-specific instructions and execution steps.

---

## Author

**Christine DeLuna**

Founder & CEO, AuVentures Health

Master of Science in Artificial Intelligence

GitHub: https://github.com/christinedeluna

Website: https://www.auventureshealth.org