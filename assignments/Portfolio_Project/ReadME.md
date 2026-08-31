# CSC580 Portfolio Project

## Deep Learning – Colorado State University Global

This directory contains my portfolio work for **CSC580 – Deep Learning**, completed as part of my **Master of Science in Artificial Intelligence at Colorado State University Global**.

The portfolio reflects two areas I explored during the course: evaluating model performance and implementing sequence-to-sequence learning with an encoder-decoder architecture.

---

## Portfolio Components

### 1. Model Performance

The `Model_Performance` directory contains earlier portfolio work focused on evaluating and improving neural network performance.

This portion of the project explores how model architecture, training configuration, and techniques such as early stopping affect model accuracy and generalization.

---

### 2. Encoder-Decoder Model for Sequence-to-Sequence Prediction

The `Encoder_Decoder_Final` directory contains my final portfolio project.

The project implements an **encoder-decoder Long Short-Term Memory (LSTM) neural network** using TensorFlow and Keras.

The model receives a sequence of six randomly generated integers and learns to return the first three integers in reverse order.

Example:

```text
Input:     [13, 28, 18, 7, 9, 5]
Target:    [18, 28, 13]
Prediction:[18, 28, 13]