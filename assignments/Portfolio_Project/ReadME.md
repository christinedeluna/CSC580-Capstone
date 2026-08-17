# CSC580 – Portfolio Project

## Encoder-Decoder Model for Sequence-to-Sequence Prediction

This directory contains the milestones, research, experiments, and final implementation for my **CSC580 – Deep Learning Portfolio Project** as part of my **Master of Science in Artificial Intelligence at Colorado State University Global**.

The final project focuses on developing an **encoder-decoder Long Short-Term Memory (LSTM) neural network for sequence-to-sequence prediction using TensorFlow and Keras**. Rather than treating the final model as a standalone assignment, this repository documents the development process through several portfolio checkpoints covering neural-network performance, model evaluation, research, experimentation, and the final encoder-decoder implementation.

## Project Progression

### Milestone 1 – TensorFlow Model Performance and Quality

**Status:** Completed

The first portfolio milestone builds upon the TensorFlow regression model developed earlier in CSC580 using the Auto MPG dataset.

The original neural network was configured to train for 1,000 epochs. Although training error continued to decrease, validation performance stabilized considerably earlier. The model was updated to use TensorFlow's `EarlyStopping` callback to monitor validation loss and automatically terminate training when additional epochs no longer produced meaningful improvement.

The updated model:

* Implemented Early Stopping with a patience of 10 epochs.
* Restored the model weights associated with the best validation performance.
* Stopped training after 92 epochs rather than completing all 1,000 epochs.
* Achieved a test Mean Absolute Error (MAE) of approximately **2.04 MPG**.
* Achieved a test Mean Squared Error (MSE) of approximately **7.19**.
* Evaluated prediction quality using actual versus predicted MPG.
* Examined the distribution of prediction errors to identify model bias and outliers.

This milestone demonstrates the importance of evaluating validation and test performance rather than assuming that additional training automatically produces a better neural network.

### Future Portfolio Milestones

Additional checkpoints will be added to this repository as the portfolio project progresses. These milestones will build toward the final encoder-decoder sequence-to-sequence model and document changes in architecture, training, evaluation, and model performance.

## Final Portfolio Project

The final implementation will use an **encoder-decoder LSTM architecture** to solve a scalable sequence-to-sequence prediction problem.

The model will receive an integer sequence containing six elements and learn to generate a three-element output sequence consisting of the first three input values in reverse order.

For example:

```text
Input:
[13, 28, 18, 7, 9, 5]

Expected Output:
[18, 28, 13]
```

The final architecture will contain:

* An LSTM encoder for processing the input sequence.
* Encoder hidden and cell states representing the source sequence.
* An LSTM decoder initialized using the encoder states.
* Teacher forcing during model training.
* Separate encoder and decoder inference models.
* Recursive sequence generation during prediction.
* One-hot encoded sequence representations.
* Evaluation using 100 newly generated sequences.

## Final Deliverables

The completed portfolio project will include:

* A research analysis examining at least four distinct industry applications of encoder-decoder models.
* A comprehensive neural-network architecture flowchart.
* A thoroughly documented Python implementation.
* Runtime prediction output.
* Model evaluation results.
* Supporting screenshots and visualizations.
* A detailed written analysis of the model and results.

## Healthcare and AuVentures Relevance

Although the portfolio exercises use educational datasets and sequence-prediction problems, the concepts explored throughout the project are relevant to my broader work developing **AuVentures**, an AI-assisted platform focused on longitudinal clinical reasoning and complex patient information.

Encoder-decoder architectures are particularly interesting in this context because they demonstrate how neural networks can transform one sequence of information into another while retaining relevant information from earlier observations. More broadly, the portfolio provides practical experience with model generalization, overfitting, validation, error analysis, sequential information, and neural-network architecture.

These concepts contribute to my longer-term goal of exploring AI systems capable of recognizing patterns across longitudinal patient information and supporting clinical reasoning while maintaining appropriate attention to model reliability, uncertainty, and ethical AI.

## Repository Structure

```text
Portfolio_Project/
│
├── Milestones/
│   └── Model_Performance/
│       ├── tensorflow_model_performance.py
│       ├── screenshots/
│       └── analysis/
│
├── Encoder_Decoder/
│   ├── encoder_decoder_model.py
│   ├── output/
│   └── screenshots/
│
├── Research/
│
├── Final_Deliverables/
│
└── README.md
```

This structure will continue to evolve as additional portfolio milestones are completed.

## Technologies

The portfolio project uses:

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* PyCharm
* Git and GitHub

---

*CSC580 – Deep Learning | Master of Science in Artificial Intelligence | Colorado State University Global*
