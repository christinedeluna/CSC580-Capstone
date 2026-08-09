# CTA3 – TensorFlow Regression with the Auto MPG Dataset

## Overview

This project demonstrates the development of a deep learning regression model using TensorFlow and Keras to predict automobile fuel efficiency (MPG). The model is trained using the UCI Auto MPG dataset and follows the complete machine learning workflow from data preparation through model evaluation.

This assignment was completed for **CSC580 – Deep Learning** in the Master of Science in Artificial Intelligence program at Colorado State University Global.

---

## Objective

Develop a regression neural network capable of predicting miles per gallon (MPG) from automobile characteristics including:

- Cylinders
- Displacement
- Horsepower
- Weight
- Acceleration
- Model Year
- Origin

---

## Project Workflow

1. Download the Auto MPG dataset
2. Import the dataset into Pandas
3. Clean missing values
4. Split data into training and testing sets
5. Perform exploratory data analysis
6. Normalize the input features
7. Build a TensorFlow/Keras regression model
8. Train the model for 1,000 epochs
9. Evaluate model performance using MAE and MSE
10. Visualize the learning process

---

## Neural Network Architecture

| Layer | Configuration |
|--------|---------------|
| Input | 7 normalized features |
| Hidden Layer 1 | Dense (64 neurons, ReLU) |
| Hidden Layer 2 | Dense (64 neurons, ReLU) |
| Output Layer | Dense (1 neuron, Linear) |

Optimizer:
- RMSprop

Loss Function:
- Mean Squared Error (MSE)

Evaluation Metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

---

## Technologies

- Python 3.11
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Repository Contents

```
CTA3-AutoMPG-Regression/
│
├── auto_mpg_regression.py
├── figures/
│   ├── Figure1_DatasetTail.png
│   ├── Figure2_Pairplot.png
│   ├── Figure3_TrainingStatistics.png
│   ├── Figure4_ModelSummary.png
│   ├── Figure5_TrainingHistory.png
│   ├── Figure6_MAE.png
│   └── Figure7_MSE.png
├── paper/
│   └── CSC580_CTA3_AutoMPG_Regression.docx
└── README.md
```

---

## Dataset

Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository: Auto MPG Dataset.*

https://archive.ics.uci.edu/ml/datasets/auto+mpg

---

## Key Learning Outcomes

This project demonstrates:

- Regression using deep learning
- Data preprocessing and normalization
- Neural network design with TensorFlow/Keras
- Model training and evaluation
- Performance visualization
- End-to-end machine learning workflow

---

## Future Applications

Although this project predicts automobile fuel efficiency, the workflow is directly transferable to other regression problems. Similar techniques could be applied to healthcare by predicting continuous clinical outcomes from longitudinal patient data. Future work may incorporate temporal modeling techniques such as Dynamic Time Warping (DTW), recurrent neural networks, or transformer-based models to support predictive analytics for complex medical conditions.

---

## Author

**Christine DeLuna**

Founder & CEO, AuVentures Health

Master of Science in Artificial Intelligence

Colorado State University Global