# CSC580 – Deep Learning

This repository contains coursework, programming assignments, portfolio milestones, and experiments completed for **CSC580 – Deep Learning** as part of my **Master of Science in Artificial Intelligence** at **Colorado State University Global**.

The course explores the practical implementation of deep learning using TensorFlow and Keras, including neural network architecture, supervised learning, optimization, hyperparameter tuning, model evaluation, convolutional neural networks, recurrent neural networks, and sequence-to-sequence learning.

Across the course, I explored not only how to build increasingly complex neural networks, but also when those architectures are appropriate. A recurring finding throughout the coursework was that greater model complexity does not automatically produce better results. Architecture selection should instead reflect the structure of the data, the problem being solved, and performance on previously unseen data.

---

## Repository Structure

```text
CSC580/
├── assignments/
│   ├── CTA1-LinearRegression/
│   ├── CTA2-NeuralNetwork/
│   ├── CTA3-AutoMPG-Regression/
│   ├── CTA4-Toxicology-Classification/
│   ├── CTA5-NN-Accuracy/
│   ├── CTA6-CNN-Cats-Dogs/
│   └── Portfolio_Project/
│       ├── Model_Performance/
│       ├── Encoder_Decoder_Final/
│       │   ├── encoder_decoder_lstm.py
│       │   ├── runtime_predictions.txt
│       │   ├── figures/
│       │   └── paper/
│       └── README.md
├── experiments/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Coursework

## CTA1 – Linear Regression

Developed a supervised machine learning model using Scikit-learn's `LinearRegression` to learn relationships between generated input features and continuous target values.

### Topics Covered

- Linear regression
- Feature generation
- Model fitting
- Prediction
- Regression evaluation

---

## CTA2 – Neural Network Revenue Prediction

Designed, trained, evaluated, and deployed a fully connected neural network using TensorFlow and Keras to predict future video game revenue from historical sales data.

### Workflow

- Data preprocessing with MinMaxScaler
- Neural network design
- Model training
- Model evaluation using Mean Squared Error (MSE)
- Model serialization
- Prediction on unseen data

---

## CTA3 – TensorFlow Regression with the Auto MPG Dataset

Built a deep learning regression model using TensorFlow and Keras to predict automobile fuel efficiency (MPG) from vehicle characteristics contained in the UCI Auto MPG dataset.

### Workflow

- Data acquisition and preprocessing
- Exploratory data analysis
- Feature normalization
- Neural network construction
- Model training
- MAE and MSE evaluation
- Training-performance visualization
- Prediction on unseen test data

This assignment later became the foundation for the first Portfolio Project milestone, where the original training process was revisited using early stopping.

---

## CTA4 – Toxicology Classification with Deep Learning

Designed and implemented a feedforward neural network for binary toxicology classification using the **DeepChem Tox21** benchmark dataset.

The project also required modernizing legacy TensorFlow 1.x instructional code for compatibility with **TensorFlow 2.21**, resolving DeepChem and RDKit dependency issues, implementing mini-batch training with dropout regularization, and evaluating model performance on previously unseen molecular compounds.

### Workflow

- DeepChem Tox21 dataset loading and preprocessing
- Molecular fingerprint feature extraction
- Binary toxicity classification
- Feedforward neural network construction
- ReLU activation
- Dropout regularization
- Binary cross-entropy loss
- Adam optimization
- Mini-batch gradient descent
- TensorBoard convergence analysis
- Validation accuracy: **90.92%**
- TensorFlow and RDKit compatibility modernization

---

## CTA5 – Neural Network Accuracy and Hyperparameter Optimization

Extended the Tox21 work from CTA4 to investigate how neural network architecture and training hyperparameters affect model performance and generalization.

The assignment compared a traditional **Random Forest classifier** with a configurable TensorFlow/Keras neural network and systematically evaluated alternative hyperparameter configurations.

### Hyperparameter Search

The initial optimization evaluated **16 neural network configurations** across three random seeds, resulting in **48 training runs**. A second refinement stage evaluated **12 additional configurations** across three seeds, adding another **36 training runs**.

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

### Results

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| Random Forest | **72.32%** | **71.97%** |
| Baseline Neural Network | 62.43% | — |
| Optimized NN – 3-Seed Average | 66.13% | — |
| Final Optimized NN | 63.68% | 65.00% |

Hyperparameter optimization improved the neural network's average validation performance from **62.43% to 66.13%**. However, the final neural network achieved **99.62% training accuracy but only 65.00% test accuracy**, indicating substantial overfitting.

The comparatively traditional Random Forest classifier generalized better than the optimized neural network. This became one of the most important findings of the course: **greater model complexity does not inherently produce better predictive performance**.

---

## CTA6 – Convolutional Neural Network Image Classification

Implemented a **Convolutional Neural Network (CNN)** for binary image classification using cat and dog images.

Unlike earlier assignments that relied on structured features such as vehicle characteristics or molecular fingerprints, this project worked directly with image data. The CNN learned hierarchical visual representations from pixel relationships, providing a practical demonstration of why specialized deep-learning architectures can be particularly effective when the structure of the underlying data supports their use.

The assignment specified Kaggle's **Dogs vs. Cats** dataset. During project setup, legacy Kaggle competition access repeatedly redirected downloads to the competition rules page, while authenticated KaggleHub access returned an HTTP 403 response. The project therefore used TensorFlow's filtered distribution of the same Dogs vs. Cats images, which excludes a small number of corrupted source files.

### Workflow

- Image dataset acquisition and inspection
- Image resizing and preprocessing
- Training and validation dataset creation
- Pixel normalization
- Convolutional feature extraction
- Pooling and spatial representation
- Binary cat-versus-dog classification
- CNN training and validation
- Accuracy and loss visualization
- Generalization analysis
- Prediction on previously unseen images

CTA6 provided an important contrast with CTA5. CTA5 demonstrated that a more complex neural network was not necessarily superior to a Random Forest for structured molecular features. CTA6 demonstrated the complementary lesson: **some data structures provide a strong architectural justification for deep learning**. Convolution is particularly appropriate when local spatial relationships contain meaningful information.

**Status: Complete**

---

# Portfolio Project – Encoder-Decoder Model for Sequence-to-Sequence Prediction

The CSC580 Portfolio Project brings together concepts developed throughout the course through the design, implementation, and evaluation of an **encoder-decoder Long Short-Term Memory (LSTM) neural network for sequence-to-sequence prediction**.

The portfolio contains both an earlier model-performance milestone and the final encoder-decoder project.

---

## Portfolio Milestone – Improving TensorFlow Model Performance and Quality

The first portfolio milestone revisited the Auto MPG regression model from CTA3 to examine model generalization and improve the training process.

The original model allowed training for up to 1,000 epochs even though validation performance stabilized much earlier. TensorFlow's `EarlyStopping` callback was introduced to monitor validation loss and automatically terminate training when additional epochs no longer produced meaningful improvement.

### Results

- Early stopping patience: **10 epochs**
- Original maximum training length: **1,000 epochs**
- Updated model stopped after: **92 epochs**
- Test Mean Absolute Error (MAE): **2.04 MPG**
- Test Mean Squared Error (MSE): **7.19**
- Actual-versus-predicted MPG evaluation
- Prediction-error distribution analysis
- Generalization evaluation on unseen test data

This milestone demonstrated that additional training does not necessarily improve a neural network. Monitoring validation performance allowed training to terminate substantially earlier while maintaining strong performance on unseen data.

---

## Final Portfolio – Encoder-Decoder LSTM

The final Portfolio Project implements an **encoder-decoder LSTM architecture using TensorFlow and Keras**.

The model receives a six-element integer sequence and learns to generate a three-element output sequence containing the first three input values in reverse order.

For example:

```text
Input:
[13, 28, 18, 7, 9, 5]

Target:
[18, 28, 13]
```

Although intentionally controlled, the problem demonstrates the core mechanics of sequence-to-sequence learning used in much more complex systems.

### Architecture

The encoder processes the complete source sequence and represents it through its **hidden state and cell state**. These states initialize the decoder, which generates the output sequence.

Training uses a shifted target sequence beginning with a start token. During inference, separate encoder and decoder models generate predictions recursively, allowing each decoder output and updated internal state to contribute to the next prediction.

### Implementation

The final project includes:

- Synthetic sequence generation
- One-hot encoded representations
- Encoder and decoder LSTM networks
- 128 LSTM units
- Hidden-state and cell-state transfer
- Shifted decoder inputs
- Teacher-forced training
- Dense softmax output
- Separate training and inference models
- Recursive decoder prediction
- Evaluation on 100 unseen sequences
- Exact-sequence accuracy
- Human-readable runtime predictions
- Runtime prediction output file
- Neural-network processing flowchart
- Research and analysis of real-world encoder-decoder applications

### Results

The trained model achieved approximately **99% validation accuracy**.

Using the stricter exact-sequence evaluation, in which all three generated values must match the expected target, independent runs achieved approximately **96–98% exact-sequence accuracy on 100 previously unseen sequences**.

Sample output:

```text
X=[2, 20, 17, 5, 24, 31]
y=[17, 20, 2]
yhat=[17, 20, 2]
```

The distinction between categorical accuracy and exact-sequence accuracy became an important part of the analysis. A model can correctly predict most individual elements while still making an error somewhere within a complete generated sequence. Exact-sequence evaluation therefore provides a more demanding measure of whether the model learned the intended transformation.

### Industry Research

The accompanying research examines encoder-decoder applications across four distinct areas:

- **Machine translation** – transforming sequences between languages
- **Healthcare** – learning from longitudinal clinical histories to support prediction of future clinical events
- **Transportation** – using historical movement sequences to predict future trajectories
- **Energy and utilities** – using historical temporal patterns for multistep demand forecasting

Despite the differences between these industries, each application involves a related computational problem: **using information contained in one sequence to generate or predict another sequence**.

**Status: Complete**

---

# Technologies

- Python 3.11
- TensorFlow 2.21
- Keras
- DeepChem
- RDKit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- TensorBoard
- Kaggle / KaggleHub
- Git and GitHub

---

# Skills Demonstrated

- Deep learning with TensorFlow and Keras
- Feedforward neural networks
- Convolutional neural networks
- Recurrent neural networks
- LSTM architecture
- Encoder-decoder architecture
- Sequence-to-sequence modeling
- Teacher forcing
- Recursive sequence inference
- Computer vision
- Image preprocessing and classification
- Regression modeling
- Binary classification
- Random Forest classification
- Hyperparameter optimization
- Multi-seed model evaluation
- Model selection and comparison
- Early stopping
- Overfitting analysis
- Validation and test-set evaluation
- Exact-sequence evaluation
- Model generalization analysis
- Data preprocessing and normalization
- Molecular fingerprint analysis
- Mini-batch gradient descent
- Dropout regularization
- Class weighting
- Adam and RMSprop optimization
- TensorBoard visualization
- TensorFlow 1.x to TensorFlow 2.x modernization
- Scientific Python dependency management

---

# Course Takeaways

One of the strongest lessons from CSC580 was that selecting an appropriate model is more important than simply selecting the most sophisticated model available.

The coursework provided several useful contrasts:

**CTA5** demonstrated that a Random Forest could generalize better than a systematically optimized neural network when working with structured molecular data.

**CTA6** demonstrated why CNNs become valuable when spatial relationships within raw image data contain meaningful information.

**The Portfolio Project** demonstrated why recurrent and encoder-decoder architectures are useful when the order and relationship between observations are themselves part of the problem.

Together, these projects reinforced a broader principle: **the architecture should follow the problem and the data, not the other way around.**

---

# Future Research

While these assignments use educational and benchmark datasets, the underlying methods translate to more complex real-world prediction and reasoning problems.

My long-term research focuses on applying artificial intelligence to complex healthcare challenges through **AuVentures Health**, particularly where longitudinal patient records, wearable data, and temporal patterns may provide information that is difficult to identify from isolated clinical observations.

CSC580 reinforced that healthcare AI should not automatically default to the newest or most complex architecture. Depending on the problem, useful systems may combine traditional machine learning, deep learning, temporal analysis, retrieval, and patient-specific modeling.

Potential approaches include **Random Forests, gradient-boosted models, convolutional neural networks, Dynamic Time Warping (DTW), recurrent neural networks, encoder-decoder architectures, transformers, retrieval-augmented generation (RAG), multimodal learning, longitudinal pattern recognition, and patient-specific modeling**.

The appropriate method should ultimately be determined by the characteristics of the data, the clinical question being addressed, generalization performance, interpretability requirements, and the consequences of prediction errors.

The objective is not to deploy the most sophisticated form of AI available, but to identify the computational approach that most reliably and responsibly solves the problem.

---

# Running the Projects

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Each assignment and portfolio component contains its own Python scripts, documentation, results, and supporting materials. Refer to the individual project directories for project-specific execution instructions.

---

# Author

**Christine DeLuna**

Founder & CEO, AuVentures Health  
Master of Science in Artificial Intelligence  
Colorado State University Global

GitHub: https://github.com/christinedeluna  
Website: https://www.auventureshealth.org