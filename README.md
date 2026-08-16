# CSC580 – Deep Learning

This repository contains coursework, programming assignments, and experiments completed for **CSC580 – Deep Learning** as part of my **Master of Science in Artificial Intelligence** at **Colorado State University Global**.

The course explores the practical implementation of deep learning using TensorFlow and Keras, including neural network architecture, supervised learning, optimization, model evaluation, and predictive analytics. Each assignment builds toward a deeper understanding of designing, training, evaluating, and deploying neural networks while applying those concepts to increasingly complex real-world datasets.

---

## Repository Structure

```text
CSC580/
├── assignments/
│   ├── CTA1-LinearRegression/
│   ├── CTA2-NeuralNetwork/
│   ├── CTA3-AutoMPG-Regression/
│   └── CTA4-Toxicology-Classification/
├── experiments/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Coursework

### CTA1 – Linear Regression

Developed a supervised machine learning model using Scikit-learn's `LinearRegression` to learn relationships between generated input features and continuous target values.

**Topics Covered**

- Linear regression
- Feature generation
- Model fitting
- Prediction
- Regression evaluation

---

### CTA2 – Neural Network Revenue Prediction

Designed, trained, evaluated, and deployed a fully connected neural network using TensorFlow and Keras to predict future video game revenue from historical sales data.

**Workflow**

- Data preprocessing with MinMaxScaler
- Neural network design
- Model training (50 epochs)
- Model evaluation using Mean Squared Error (MSE)
- Model serialization
- Predicting unseen data

---

### CTA3 – TensorFlow Regression with the Auto MPG Dataset

Built a deep learning regression model using TensorFlow and Keras to predict automobile fuel efficiency (MPG) from vehicle characteristics contained in the UCI Auto MPG dataset.

**Workflow**

- Downloading and importing data
- Data cleaning and preprocessing
- Exploratory data analysis
- Feature normalization
- Neural network construction
- Model training (1,000 epochs)
- Model evaluation using MAE and MSE
- Visualization of training performance

---

### CTA4 – Toxicology Classification with Deep Learning

Designed and implemented a feedforward neural network for binary toxicology classification using the **DeepChem Tox21** benchmark dataset. This project modernized legacy TensorFlow 1.x instructional code for compatibility with **TensorFlow 2.21**, resolved DeepChem and RDKit dependency issues, implemented mini-batch gradient descent with dropout regularization, monitored model convergence using TensorBoard, and evaluated model performance on previously unseen molecular compounds.

**Workflow**

- Loading and preprocessing the DeepChem Tox21 dataset
- Molecular fingerprint feature extraction (1,024 features)
- Binary toxicity classification
- Feedforward neural network construction
- ReLU activation and dropout regularization
- Binary cross-entropy loss
- Adam optimization
- Mini-batch gradient descent
- TensorBoard visualization and model convergence analysis
- Validation accuracy: **90.92%**
- Technical documentation of TensorFlow and RDKit compatibility modernization

---

## Technologies

- Python 3.11
- TensorFlow 2.21
- Keras
- DeepChem
- RDKit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- TensorBoard

---

## Skills Demonstrated

- Deep learning with TensorFlow
- Feedforward neural network architecture
- Binary classification
- Regression modeling
- Data preprocessing and normalization
- Molecular fingerprint analysis
- Mini-batch gradient descent
- Dropout regularization
- Adam optimization
- TensorBoard visualization
- Model convergence analysis
- Performance evaluation
- Scientific Python dependency management
- Biomedical machine learning workflows

---

## Future Research

While these assignments use educational datasets, the underlying methods directly translate to real-world prediction problems.

My long-term research focuses on applying artificial intelligence to complex healthcare challenges through **AuVentures Health**. Specifically, I am interested in combining deep learning, longitudinal patient records, wearable sensor data, and temporal analysis methods to identify patterns that precede autoimmune and neurodevelopmental disease flares. Although future clinical systems will likely incorporate more advanced techniques such as Dynamic Time Warping (DTW), recurrent neural networks, transformers, retrieval-augmented generation (RAG), and multimodal learning, the concepts explored throughout this course provide the foundational building blocks for those future applications.

---

## Running the Projects

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Each assignment contains its own Python scripts, documentation, and supporting materials. Refer to the individual assignment folders for project-specific execution instructions and research papers.

---

## Author

**Christine DeLuna**

Founder & CEO, AuVentures Health

Master of Science in Artificial Intelligence

Colorado State University Global

GitHub: https://github.com/christinedeluna

Website: https://www.auventureshealth.org