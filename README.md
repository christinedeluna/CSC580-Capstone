# CSC580 – Deep Learning

This repository contains coursework, programming assignments, portfolio milestones, and experiments completed for **CSC580 – Deep Learning** as part of my **Master of Science in Artificial Intelligence** at **Colorado State University Global**.

The course explores the practical implementation of deep learning using TensorFlow and Keras, including neural network architecture, supervised learning, optimization, model evaluation, recurrent neural networks, and predictive analytics. Each assignment builds toward a deeper understanding of designing, training, evaluating, and improving neural networks while applying those concepts to increasingly complex datasets and sequence-based problems.

---

## Repository Structure

```text
CSC580/
├── assignments/
│   ├── CTA1-LinearRegression/
│   ├── CTA2-NeuralNetwork/
│   ├── CTA3-AutoMPG-Regression/
│   ├── CTA4-Toxicology-Classification/
│   └── Portfolio_Project/
│       ├── Milestones/
│       ├── Encoder_Decoder/
│       ├── Research/
│       ├── Final_Deliverables/
│       └── README.md
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

* Linear regression
* Feature generation
* Model fitting
* Prediction
* Regression evaluation

---

### CTA2 – Neural Network Revenue Prediction

Designed, trained, evaluated, and deployed a fully connected neural network using TensorFlow and Keras to predict future video game revenue from historical sales data.

**Workflow**

* Data preprocessing with MinMaxScaler
* Neural network design
* Model training (50 epochs)
* Model evaluation using Mean Squared Error (MSE)
* Model serialization
* Predicting unseen data

---

### CTA3 – TensorFlow Regression with the Auto MPG Dataset

Built a deep learning regression model using TensorFlow and Keras to predict automobile fuel efficiency (MPG) from vehicle characteristics contained in the UCI Auto MPG dataset.

**Workflow**

* Downloading and importing data
* Data cleaning and preprocessing
* Exploratory data analysis
* Feature normalization
* Neural network construction
* Model training (1,000 epochs)
* Model evaluation using MAE and MSE
* Visualization of training performance

---

### CTA4 – Toxicology Classification with Deep Learning

Designed and implemented a feedforward neural network for binary toxicology classification using the **DeepChem Tox21** benchmark dataset. This project modernized legacy TensorFlow 1.x instructional code for compatibility with **TensorFlow 2.21**, resolved DeepChem and RDKit dependency issues, implemented mini-batch gradient descent with dropout regularization, monitored model convergence using TensorBoard, and evaluated model performance on previously unseen molecular compounds.

**Workflow**

* Loading and preprocessing the DeepChem Tox21 dataset
* Molecular fingerprint feature extraction (1,024 features)
* Binary toxicity classification
* Feedforward neural network construction
* ReLU activation and dropout regularization
* Binary cross-entropy loss
* Adam optimization
* Mini-batch gradient descent
* TensorBoard visualization and model convergence analysis
* Validation accuracy: **90.92%**
* Technical documentation of TensorFlow and RDKit compatibility modernization

---

## Portfolio Project – Encoder-Decoder Model for Sequence-to-Sequence Prediction

The CSC580 Portfolio Project brings together concepts developed throughout the course and applies them toward the design and evaluation of an **encoder-decoder Long Short-Term Memory (LSTM) neural network for sequence-to-sequence prediction**.

Rather than treating the final model as a single implementation, the `Portfolio_Project` directory documents the project incrementally through milestones, research, model evaluation, experimentation, and the final encoder-decoder implementation.

### Portfolio Milestone 1 – Improving TensorFlow Model Performance and Quality

The first portfolio milestone revisited the Auto MPG regression model developed in CTA3 to examine model generalization and improve the training process.

The original model trained for 1,000 epochs even though validation performance stabilized much earlier. TensorFlow's `EarlyStopping` callback was introduced to monitor validation loss and automatically terminate training when additional epochs no longer produced meaningful improvement.

**Results**

* Early Stopping patience: **10 epochs**
* Original maximum training length: **1,000 epochs**
* Updated model stopped after: **92 epochs**
* Test Mean Absolute Error (MAE): **2.04 MPG**
* Test Mean Squared Error (MSE): **7.19**
* Evaluated actual versus predicted MPG
* Analyzed prediction-error distribution
* Examined model generalization using previously unseen test data

This milestone demonstrated that additional training does not necessarily improve a neural network. Monitoring validation performance allowed the model to stop substantially earlier while maintaining strong performance on unseen data.

### Final Encoder-Decoder Model

The final Portfolio Project will implement an encoder-decoder LSTM architecture using TensorFlow and Keras.

The sequence-to-sequence problem will provide the model with a six-element integer sequence and require it to generate a three-element sequence containing the first three input values in reverse order.

For example:

```text
Input:
[13, 28, 18, 7, 9, 5]

Target:
[18, 28, 13]
```

The final implementation will explore:

* Encoder-decoder neural network architecture
* Long Short-Term Memory (LSTM) networks
* Sequence-to-sequence prediction
* Encoder hidden and cell states
* Teacher forcing during training
* One-hot encoded sequence representations
* Separate training and inference models
* Recursive decoder prediction
* Model generalization and sequence accuracy
* Neural-network architecture visualization

Additional portfolio milestones will be documented as the project progresses.

---

## Technologies

* Python 3.11
* TensorFlow 2.21
* Keras
* DeepChem
* RDKit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* TensorBoard
* Git and GitHub

---

## Skills Demonstrated

* Deep learning with TensorFlow and Keras
* Feedforward neural network architecture
* Regression modeling
* Binary classification
* Model training and optimization
* Early stopping and overfitting mitigation
* Validation and test-set evaluation
* Model generalization analysis
* Prediction-error analysis
* Data preprocessing and normalization
* Molecular fingerprint analysis
* Mini-batch gradient descent
* Dropout regularization
* Adam and RMSprop optimization
* TensorBoard visualization
* Model convergence analysis
* Scientific Python dependency management
* Biomedical machine learning workflows
* Recurrent neural networks
* LSTM architecture
* Sequence-to-sequence modeling

---

## Future Research

While these assignments use educational and benchmark datasets, the underlying methods directly translate to more complex real-world prediction and reasoning problems.

My long-term research focuses on applying artificial intelligence to complex healthcare challenges through **AuVentures Health**. Specifically, I am interested in combining deep learning, longitudinal patient records, wearable sensor data, and temporal analysis methods to identify patterns that precede autoimmune and neurodevelopmental disease flares.

The Portfolio Project is particularly relevant to this research direction. Early stopping and test-set evaluation reinforce the importance of developing models that generalize beyond their training data, while encoder-decoder and LSTM architectures introduce methods for learning from sequential information. These concepts provide foundations for exploring models that can interpret longitudinal patient histories, retain clinically relevant temporal context, and transform sequences of patient information into useful representations for clinical reasoning support.

Future clinical systems may incorporate techniques such as **Dynamic Time Warping (DTW), recurrent neural networks, transformers, retrieval-augmented generation (RAG), multimodal learning, longitudinal pattern recognition, and patient-specific reasoning models**. The concepts explored throughout CSC580 provide foundational building blocks for investigating these more advanced architectures while maintaining an emphasis on model reliability, generalization, and ethical AI.

---

## Running the Projects

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Each assignment and portfolio milestone contains its own Python scripts, documentation, and supporting materials. Refer to the individual project folders for project-specific execution instructions, results, screenshots, and research papers.

---

## Author

**Christine DeLuna**

Founder & CEO, AuVentures Health

Master of Science in Artificial Intelligence

Colorado State University Global

GitHub: https://github.com/christinedeluna

Website: https://www.auventureshealth.org
