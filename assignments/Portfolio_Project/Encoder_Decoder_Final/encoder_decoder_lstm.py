"""
CSC580 - Deep Learning
Portfolio Project: Encoder-Decoder Model for Sequence-to-Sequence Prediction

This program implements an encoder-decoder LSTM using Keras/TensorFlow.

The model receives a sequence of six integers and learns to return
the first three integers in reverse order.

Example:

Input:
[13, 28, 18, 7, 9, 5]

Target:
[18, 28, 13]

The program demonstrates:
1. Sequence generation
2. One-hot encoding
3. Encoder-decoder LSTM construction
4. Model training
5. Separate encoder/decoder inference
6. Evaluation on 100 unseen sequences
"""


# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================
# randint is used to generate random integer sequences.
#
# NumPy is used for arrays, decoding one-hot vectors, and
# constructing the start-of-sequence input.
#
# TensorFlow/Keras provides the neural network components.

from random import randint

import numpy as np
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


# ============================================================
# SECTION 2: DEFINE THE ENCODER-DECODER MODELS
# ============================================================
# Three related models are created:
#
# 1. Training model
#    Encoder -> Decoder -> Output
#
# 2. Inference encoder
#    Used to encode a new source sequence.
#
# 3. Inference decoder
#    Used to generate predictions one step at a time.
#
# The encoder passes two LSTM states to the decoder:
#
# state_h = hidden state
# state_c = cell state
#
# These states contain the encoder's learned representation
# of the input sequence.


def define_models(n_input, n_output, n_units):
    """
    Define the encoder-decoder LSTM models used for training
    and inference.

    Parameters:
        n_input:
            Number of possible input features at each time step.

        n_output:
            Number of possible output features at each time step.

        n_units:
            Number of LSTM units in the encoder and decoder.

    Returns:
        model:
            Complete encoder-decoder model used during training.

        encoder_model:
            Encoder used during inference.

        decoder_model:
            Decoder used during inference.
    """

    # --------------------------------------------------------
    # Training Encoder
    # --------------------------------------------------------
    # The encoder receives the source sequence.
    #
    # Example:
    # [13, 28, 18, 7, 9, 5]
    #
    # Because the values are one-hot encoded, each time step
    # actually contains a vector of 51 values.

    encoder_inputs = Input(shape=(None, n_input))

    encoder = LSTM(
        n_units,
        return_state=True
    )

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # Save the final hidden and cell states.
    # These states initialize the decoder.

    encoder_states = [state_h, state_c]

    # --------------------------------------------------------
    # Training Decoder
    # --------------------------------------------------------
    # During training, the decoder receives a shifted version
    # of the correct target sequence.
    #
    # Target:
    # [18, 28, 13]
    #
    # Decoder input:
    # [0, 18, 28]
    #
    # Zero acts as the start-of-sequence token.

    decoder_inputs = Input(shape=(None, n_output))

    decoder_lstm = LSTM(
        n_units,
        return_sequences=True,
        return_state=True
    )

    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs,
        initial_state=encoder_states
    )

    # Dense + softmax produces a probability distribution
    # across all 51 possible output values.

    decoder_dense = Dense(
        n_output,
        activation="softmax"
    )

    decoder_outputs = decoder_dense(decoder_outputs)

    # --------------------------------------------------------
    # Complete Training Model
    # --------------------------------------------------------
    # The training model receives:
    #
    # 1. Source sequence
    # 2. Shifted target sequence
    #
    # It learns to predict the actual target sequence.

    model = Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs
    )

    # --------------------------------------------------------
    # Inference Encoder
    # --------------------------------------------------------
    # Once training is finished, the encoder can be used
    # independently.
    #
    # It receives a new sequence and returns its LSTM states.

    encoder_model = Model(
        encoder_inputs,
        encoder_states
    )

    # --------------------------------------------------------
    # Inference Decoder
    # --------------------------------------------------------
    # During prediction, we do not know the correct target.
    #
    # Therefore, the decoder must generate one prediction,
    # update its states, and then generate the next prediction.

    decoder_state_input_h = Input(
        shape=(n_units,)
    )

    decoder_state_input_c = Input(
        shape=(n_units,)
    )

    decoder_states_inputs = [
        decoder_state_input_h,
        decoder_state_input_c
    ]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs,
        initial_state=decoder_states_inputs
    )

    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return model, encoder_model, decoder_model


# ============================================================
# SECTION 3: GENERATE RANDOM SOURCE SEQUENCES
# ============================================================
# The assignment uses randomly generated integer sequences.
#
# Zero is NOT included because zero is reserved as the
# start-of-sequence token for the decoder.
#
# Example:
# [13, 28, 18, 7, 9, 5]


def generate_sequence(length, n_unique):
    """
    Generate a random sequence of integers.

    Values range from 1 through n_unique - 1 because zero
    is reserved as the start-of-sequence token.
    """

    return [
        randint(1, n_unique - 1)
        for _ in range(length)
    ]


# ============================================================
# SECTION 4: CREATE THE SEQUENCE-TO-SEQUENCE DATASET
# ============================================================
# Each training example contains three sequences:
#
# X1 = Source sequence
# X2 = Decoder input
# y  = Expected target
#
# Example:
#
# X1 = [13, 28, 18, 7, 9, 5]
# X2 = [0, 18, 28]
# y  = [18, 28, 13]
#
# The target is created by taking the first three elements
# from the source and reversing them.


def get_dataset(n_in, n_out, cardinality, n_samples):
    """
    Generate source, decoder-input, and target sequences.

    All sequences are converted to one-hot encoded vectors
    before being returned.
    """

    X1 = []
    X2 = []
    y = []

    for _ in range(n_samples):

        # Generate the six-element source sequence.

        source = generate_sequence(
            n_in,
            cardinality
        )

        # Select the first three values.

        target = source[:n_out]

        # Reverse those three values.
        #
        # Example:
        # [13, 28, 18] -> [18, 28, 13]

        target.reverse()

        # Create the input used by the decoder.
        #
        # Target:
        # [18, 28, 13]
        #
        # Decoder input:
        # [0, 18, 28]

        target_in = [0] + target[:-1]

        # ----------------------------------------------------
        # One-Hot Encoding
        # ----------------------------------------------------
        # The neural network does not directly receive the
        # integer 18, for example.
        #
        # Instead, 18 becomes a vector containing 51 positions
        # where position 18 contains a 1 and all others are 0.

        src_encoded = to_categorical(
            source,
            num_classes=cardinality
        )

        tar_encoded = to_categorical(
            target,
            num_classes=cardinality
        )

        tar2_encoded = to_categorical(
            target_in,
            num_classes=cardinality
        )

        X1.append(src_encoded)
        X2.append(tar2_encoded)
        y.append(tar_encoded)

    return (
        np.array(X1),
        np.array(X2),
        np.array(y)
    )


# ============================================================
# SECTION 5: DECODE ONE-HOT ENCODED VALUES
# ============================================================
# Neural networks work with the one-hot encoded vectors,
# but humans want to see the original integer values.
#
# This function converts:
#
# one-hot vectors
#
# back into:
#
# [18, 28, 13]


def one_hot_decode(encoded_seq):
    """
    Convert a one-hot encoded sequence back into integers.
    """

    return [
        np.argmax(vector)
        for vector in encoded_seq
    ]


# ============================================================
# SECTION 6: GENERATE A PREDICTION
# ============================================================
# Training and prediction work differently.
#
# During training, the decoder receives the correct previous
# target value.
#
# During prediction, the correct target is unknown.
#
# Therefore:
#
# 1. Encode the source once.
# 2. Begin with the start token.
# 3. Predict one output.
# 4. Pass the updated LSTM states forward.
# 5. Use the prediction to generate the next output.
# 6. Repeat until three values have been generated.


def predict_sequence(
        infenc,
        infdec,
        source,
        n_steps,
        cardinality
):
    """
    Generate a target sequence from a new source sequence
    using the trained inference encoder and decoder.
    """

    # Encode the source sequence.
    #
    # The returned values are the encoder's hidden state
    # and cell state.

    state = infenc.predict(
        source,
        verbose=0
    )

    # Create the initial decoder input.
    #
    # This is a one-hot-sized vector initialized to zeros,
    # representing the start-of-sequence token.

    target_seq = np.zeros(
        (1, 1, cardinality)
    )

    output = []

    # Generate the three output values one at a time.

    for _ in range(n_steps):

        yhat, h, c = infdec.predict(
            [target_seq] + state,
            verbose=0
        )

        # Save the predicted probability distribution.

        output.append(
            yhat[0, 0, :]
        )

        # Update the decoder's internal states.

        state = [h, c]

        # Use the current prediction as the input
        # for the next decoder step.
        #
        # This follows the inference procedure supplied
        # in the assignment.

        target_seq = yhat

    return np.array(output)


# ============================================================
# SECTION 7: MAIN PROGRAM
# ============================================================
# Everything below this point actually runs the experiment.
#
# The functions above define HOW the experiment works.
# This section tells Python WHEN to perform each step.


if __name__ == "__main__":

    # ========================================================
    # 7A. CONFIGURE THE ASSIGNMENT
    # ========================================================
    #
    # The assignment specifies:
    #
    # 50 possible integer values
    # + 1 reserved start token
    #
    # Six input time steps
    # Three output time steps

    n_features = 50 + 1
    n_steps_in = 6
    n_steps_out = 3

    # ========================================================
    # 7B. VERIFY DATASET GENERATION
    # ========================================================
    # Generate one example before training so that we can
    # verify the dataset is constructed correctly.

    X1, X2, y = get_dataset(
        n_steps_in,
        n_steps_out,
        n_features,
        1
    )

    print("Shapes:")
    print("X1:", X1.shape)
    print("X2:", X2.shape)
    print("y: ", y.shape)

    print("\nDecoded sequences:")
    print(
        "X1 =",
        one_hot_decode(X1[0])
    )

    print(
        "X2 =",
        one_hot_decode(X2[0])
    )

    print(
        "y  =",
        one_hot_decode(y[0])
    )

    # ========================================================
    # 7C. BUILD THE ENCODER-DECODER LSTM
    # ========================================================

    print(
        "\nBuilding encoder-decoder LSTM..."
    )

    # The assignment specifies 128 LSTM units.

    train, infenc, infdec = define_models(
        n_features,
        n_features,
        128
    )

    # Configure how the network learns.
    #
    # Adam updates the model weights.
    #
    # Categorical crossentropy measures the error between
    # the expected and predicted categorical outputs.

    train.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ========================================================
    # 7D. GENERATE TRAINING DATA
    # ========================================================
    # We generate 10,000 random training examples.
    #
    # This is our experimental training configuration;
    # the assignment does not prescribe the number of
    # training examples.

    X1_train, X2_train, y_train = get_dataset(
        n_steps_in,
        n_steps_out,
        n_features,
        10000
    )

    # ========================================================
    # 7E. TRAIN THE MODEL
    # ========================================================

    print("\nTraining model...")

    # 80% of the generated data is used for training.
    # 20% is reserved for validation.
    #
    # We train for 20 epochs using batches of 64 examples.

    history = train.fit(
        [X1_train, X2_train],
        y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # ========================================================
    # 7F. EVALUATE ON 100 UNSEEN SEQUENCES
    # ========================================================
    # The assignment requires predictions for 100 newly
    # generated source sequences.
    #
    # A prediction only counts as correct if ALL THREE
    # output values match the expected target sequence.

    print(
        "\nEvaluating model on 100 unseen sequences..."
    )

    total = 100
    correct = 0

    for _ in range(total):

        # Generate a new sequence that was not part
        # of the training dataset.

        X1_test, X2_test, y_test = get_dataset(
            n_steps_in,
            n_steps_out,
            n_features,
            1
        )

        # Ask the trained encoder-decoder to predict
        # the three-element target.

        prediction = predict_sequence(
            infenc,
            infdec,
            X1_test,
            n_steps_out,
            n_features
        )

        # Convert the expected and predicted values
        # back into readable integers.

        expected = one_hot_decode(
            y_test[0]
        )

        predicted = one_hot_decode(
            prediction
        )

        # Exact sequence matching:
        #
        # Expected:  [45, 2, 12]
        # Predicted: [45, 2, 12]
        #
        # All three must match.

        if expected == predicted:
            correct += 1

    # Calculate exact sequence accuracy.

    sequence_accuracy = (
        correct / total
    ) * 100

    print(
        f"Exact sequence accuracy: "
        f"{correct}/{total} "
        f"({sequence_accuracy:.2f}%)"
    )

    # ========================================================
    # 7G. GENERATE AND SAVE SAMPLE PREDICTIONS
    # ========================================================
    # Generate 10 additional unseen sequences so the model's
    # predictions can be inspected directly.
    #
    # X     = original six-element source sequence
    # y     = expected three-element target sequence
    # yhat  = model's predicted target sequence

    print("\nSample predictions:")

    prediction_lines = []

    for _ in range(10):

        # Generate a new unseen sequence
        X1_sample, X2_sample, y_sample = get_dataset(
            n_steps_in,
            n_steps_out,
            n_features,
            1
        )

        # Generate the model's prediction
        prediction = predict_sequence(
            infenc,
            infdec,
            X1_sample,
            n_steps_out,
            n_features
        )

        # Convert one-hot encoded values back to integers
        source = one_hot_decode(X1_sample[0])
        expected = one_hot_decode(y_sample[0])
        predicted = one_hot_decode(prediction)

        # Format the result for display
        result = (
            f"X={source} "
            f"y={expected} "
            f"yhat={predicted}"
        )

        print(result)
        prediction_lines.append(result)

    # Save the runtime results to a text file
    output_file = "runtime_predictions.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(
            "CSC580 - Encoder-Decoder LSTM\n"
            "Runtime Prediction Results\n"
            "========================================\n\n"
        )

        file.write(
            f"Exact sequence accuracy: "
            f"{correct}/{total} "
            f"({sequence_accuracy:.2f}%)\n\n"
        )

        file.write("Sample Predictions:\n\n")

        for line in prediction_lines:
            file.write(line + "\n")

    print(
        f"\nRuntime predictions saved to: {output_file}"
    )