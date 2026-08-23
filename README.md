# CSC580 – Deep Learning

This repository contains coursework, programming assignments, portfolio milestones, and experiments completed for **CSC580 – Deep Learning** as part of my **Master of Science in Artificial Intelligence** at **Colorado State University Global**.

The course explores the practical implementation of deep learning using TensorFlow and Keras, including neural network architecture, supervised learning, optimization, hyperparameter tuning, model evaluation, recurrent neural networks, and predictive analytics. Each assignment builds toward a deeper understanding of designing, training, evaluating, and improving neural networks while applying those concepts to increasingly complex datasets and sequence-based problems.

---

## Repository Structure

```text
CSC580/
├── assignments/
│   ├── CTA1-LinearRegression/
│   ├── CTA2-NeuralNetwork/
│   ├── CTA3-AutoMPG-Regression/
│   ├── CTA4-Toxicology-Classification/
│   ├── CTA-NN-Accuracy/
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

### CTA5 – Neural Network Accuracy and Hyperparameter Optimization

Extended the Tox21 classification work from CTA4 to investigate how neural network architecture and training hyperparameters affect model performance and generalization. The assignment compared a traditional **Random Forest classifier** with a configurable TensorFlow/Keras neural network and systematically evaluated alternative hyperparameter configurations.

The instructor-provided implementation again relied on legacy TensorFlow 1.x graph and session APIs. Building upon the compatibility lessons from CTA4, the experimental design was preserved while the implementation was modernized for **TensorFlow 2.21 and the current Keras API**.

**Workflow**

* Established a 50-estimator Random Forest baseline
* Developed a configurable TensorFlow/Keras neural network
* Evaluated hidden-layer width and network depth
* Tuned learning rate and dropout regularization
* Evaluated training duration and batch size
* Tested positive-class weighting
* Repeated configurations across multiple random seeds
* Averaged validation performance to reduce initialization variance
* Conducted a two-stage hyperparameter search
* Evaluated the final optimized model on previously unseen test data
* Analyzed model generalization and overfitting
* Compared optimized deep learning performance with traditional machine learning

**Hyperparameter Search**

The initial optimization evaluated **16 neural network configurations**, each across three random seeds, resulting in **48 training runs**. A second refinement stage evaluated **12 additional configurations** across three seeds, adding another **36 training runs**.

The strongest configuration used:

```text
Hidden Units:       50
Hidden Layers:      2
Learning Rate:      0.001
Dropout:            0.50
Epochs:             60
Batch Size:         50
Positive Weighting: Enabled
```

**Results**

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| Random Forest | **72.32%** | **71.97%** |
| Baseline Neural Network | 62.43% | — |
| Optimized NN – 3-Seed Average | 66.13% | — |
| Final Optimized NN | 63.68% | 65.00% |

Hyperparameter optimization improved the neural network's average validation performance from **62.43% to 66.13%**. However, the final neural network achieved **99.62% training accuracy but only 65.00% test accuracy**, indicating substantial overfitting.

Perhaps the most important finding was that the comparatively traditional Random Forest classifier generalized better than the optimized neural network. This experiment reinforced that **greater model complexity does not inherently produce better predictive performance**. Model selection should be driven by the characteristics of the problem, empirical performance, and generalization to unseen data rather than an assumption that deep learning will always outperform traditional machine learning.

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
* Random Forest classification
* Hyperparameter optimization
* Multi-seed model evaluation
* Model selection and comparison
* Model training and optimization
* Early stopping and overfitting mitigation
* Validation and test-set evaluation
* Model generalization analysis
* Prediction-error analysis
* Data preprocessing and normalization
* Molecular fingerprint analysis
* Mini-batch gradient descent
* Dropout regularization
* Class weighting
* Adam and RMSprop optimization
* TensorBoard visualization
* Model convergence analysis
* Scientific Python dependency management
* TensorFlow 1.x to TensorFlow 2.x modernization
* Biomedical machine learning workflows
* Recurrent neural networks
* LSTM architecture
* Sequence-to-sequence modeling

---

## Future Research

While these assignments use educational and benchmark datasets, the underlying methods directly translate to more complex real-world prediction and reasoning problems.

My long-term research focuses on applying artificial intelligence to complex healthcare challenges through **AuVentures Health**. Specifically, I am interested in combining deep learning, longitudinal patient records, wearable sensor data, and temporal analysis methods to identify patterns that precede autoimmune and neurodevelopmental disease flares.

The results throughout CSC580 have also reinforced that healthcare AI should not automatically default to the newest or most complex architecture. CTA5 demonstrated that a traditional Random Forest classifier could outperform a systematically optimized neural network on structured biomedical data. Depending on the problem, established machine learning methods may provide stronger generalization, greater interpretability, or lower computational complexity than deep learning.

The Portfolio Project extends this exploration into sequential modeling. Early stopping and test-set evaluation reinforce the importance of developing models that generalize beyond their training data, while encoder-decoder and LSTM architectures introduce methods for learning from sequential information. These concepts provide foundations for exploring models that can interpret longitudinal patient histories, retain clinically relevant temporal context, and transform sequences of patient information into useful representations for clinical reasoning support.

Future clinical systems may therefore incorporate a combination of **Random Forests, gradient-boosted models, Dynamic Time Warping (DTW), recurrent neural networks, transformers, retrieval-augmented generation (RAG), multimodal learning, longitudinal pattern recognition, and patient-specific modeling** rather than relying on a single architecture. The appropriate method should be determined by the characteristics of the data, the intended clinical question, generalization performance, interpretability requirements, and the consequences of prediction errors.

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