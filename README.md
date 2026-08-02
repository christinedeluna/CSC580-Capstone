# CSC580 – Deep Learning Capstone

This repository contains coursework, experiments, and programming assignments completed for **CSC580 – Deep Learning** as part of my Master of Science in Artificial Intelligence program at Colorado State University Global.

The objective of this course is to develop an understanding of modern deep learning techniques using TensorFlow and Keras, including neural network design, supervised learning, model evaluation, and prediction.

---

## Repository Structure

```
CSC580-Capstone/
├── assignments/
│   ├── CTA1-LinearRegression/
│   └── CTA2-NeuralNetwork/
├── experiments/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Assignments

### CTA1 – Linear Regression
Implemented a supervised machine learning model that learns an algebraic function from generated training data using Scikit-learn's `LinearRegression`.

**Topics**
- Feature generation
- Model training
- Regression analysis
- Prediction

---

### CTA2 – Neural Networks with TensorFlow/Keras

Designed, trained, evaluated, and deployed a fully connected neural network capable of predicting future video game revenue based on historical sales data.

#### Workflow

1. Data preprocessing using MinMaxScaler
2. Neural network construction using TensorFlow/Keras
3. Model training (50 epochs)
4. Model evaluation using Mean Squared Error (MSE)
5. Saving and loading trained models
6. Predicting revenue for unseen data

#### Technologies

- Python 3.11
- TensorFlow 2.x
- Keras
- Scikit-learn
- Pandas
- NumPy

---

## Skills Demonstrated

- Data preprocessing
- Feature scaling
- Supervised learning
- Regression using neural networks
- TensorFlow model development
- Model evaluation
- Saving and loading trained models
- Predictive analytics

---

## Future Applications

Although these assignments use educational datasets, the concepts directly transfer to real-world prediction problems. My long-term research interests involve applying deep learning and longitudinal patient data to identify patterns that precede disease flares in autoimmune and other complex chronic conditions, supporting clinical decision-making through AI-assisted predictive analytics.

---

## Running the Projects

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the preprocessing script:

```bash
python scale_data.py
```

Train the neural network:

```bash
python create_model.py
```

Generate predictions:

```bash
python predict.py
```

---

## Author

**Christine DeLuna**

Founder & CEO, AuVentures Health

Master of Science in Artificial Intelligence

GitHub: https://github.com/christinedeluna