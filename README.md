# CSC580 – Deep Learning

This repository contains coursework, programming assignments, and experiments completed for **CSC580 – Deep Learning** as part of my **Master of Science in Artificial Intelligence** at **Colorado State University Global**.

The course explores the practical implementation of deep learning using TensorFlow and Keras, including neural network architecture, regression, optimization, model evaluation, and predictive analytics. Each assignment builds toward a stronger understanding of designing, training, and deploying neural networks for real-world applications.

---

## Repository Structure

```
CSC580/
├── assignments/
│   ├── CTA1-LinearRegression/
│   ├── CTA2-NeuralNetwork/
│   └── CTA3-AutoMPG-Regression/
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

## Technologies

- Python 3.11
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Skills Demonstrated

- Deep learning with TensorFlow
- Neural network architecture design
- Regression modeling
- Data preprocessing and normalization
- Exploratory data analysis
- Model training and optimization
- Performance evaluation using MAE and MSE
- Predictive analytics
- Scientific visualization

---

## Future Research

While these assignments use educational datasets, the underlying methods directly translate to real-world prediction problems.

My long-term research focuses on applying artificial intelligence to complex healthcare challenges through **AuVentures Health**. Specifically, I am interested in combining deep learning, longitudinal patient records, wearable sensor data, and temporal analysis methods to identify patterns that precede autoimmune and neurodevelopmental disease flares. Although future clinical systems will likely incorporate more advanced techniques such as Dynamic Time Warping (DTW), recurrent neural networks, transformers, and multimodal learning, the concepts explored in this course provide the foundational building blocks for those future applications.

---

## Running the Projects

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Each assignment contains its own Python scripts and documentation. Refer to the individual assignment folders for execution instructions.

---

## Author

**Christine DeLuna**

Founder & CEO, AuVentures Health

Master of Science in Artificial Intelligence

Colorado State University Global

GitHub: https://github.com/christinedeluna