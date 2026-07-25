import random
from sklearn.linear_model import LinearRegression


def get_variable_count():
    """
    Prompt to enter the number of variables (4–8).
    """
    while True:
        try:
            count = int(input("Enter the number of variables (4-8): "))

            if 4 <= count <= 8:
                return count

            print("Please enter a number between 4 and 8.")

        except ValueError:
            print("Invalid input. Please enter an integer.")


def get_coefficients(variable_count):
    """
    Prompt to enter the coefficients for each variable.
    """
    coefficients = []

    print("\nEnter the coefficients for your equation:")

    for i in range(variable_count):
        coefficient = float(input(f"Coefficient for x{i + 1}: "))
        coefficients.append(coefficient)

    return coefficients


def generate_training_data(coefficients, train_count=1000, train_limit=1000):
    """
    Generate random training data based on the equation.
    """
    train_input = []
    train_output = []

    for _ in range(train_count):

        # Generate one random set of inputs
        inputs = [random.randint(0, train_limit) for _ in coefficients]

        # Calculate the correct output
        output = sum(coef * value for coef, value in zip(coefficients, inputs))

        train_input.append(inputs)
        train_output.append(output)

    return train_input, train_output


def train_model(train_input, train_output):
    """
    Train a Linear Regression model using the generated training data.
    """
    predictor = LinearRegression()

    predictor.fit(train_input, train_output)

    return predictor


def get_test_values(variable_count):
    """
    Prompt the user to enter values for testing the model.
    """
    test_values = []

    print("\nEnter values to test the model:")

    for i in range(variable_count):
        value = float(input(f"Value for x{i + 1}: "))
        test_values.append(value)

    return test_values


def calculate_actual_value(coefficients, test_values):
    """
    Calculate the actual output using the original equation.
    """
    return sum(coef * value for coef, value in zip(coefficients, test_values))


def main():
    print("Machine Learning Linear Equation Predictor")

    # Step 1: Get number of variables
    variable_count = get_variable_count()

    print(f"\nYou selected {variable_count} variables.")

    # Step 2: Get coefficients
    coefficients = get_coefficients(variable_count)

    print("\nEquation coefficients:")
    print(coefficients)

    # Step 3: Generate training data
    train_input, train_output = generate_training_data(coefficients)

    print("\nTraining data generated successfully!")
    print(f"Training samples: {len(train_input)}")

    print("\nFirst training example:")
    print(f"Inputs : {train_input[0]}")
    print(f"Output : {train_output[0]}")

    # Step 4: Train the model
    predictor = train_model(train_input, train_output)

    print("\nMachine Learning model trained successfully!")

    print("\nOriginal coefficients:")
    print(coefficients)

    print("\nLearned coefficients:")
    print(predictor.coef_)

    # Step 5: Test the model
    test_values = get_test_values(variable_count)

    predicted_value = predictor.predict([test_values])[0]

    actual_value = calculate_actual_value(coefficients, test_values)

    print("\nResults")
    print("-" * 30)
    print(f"Test values     : {test_values}")
    print(f"Predicted value : {predicted_value:.2f}")
    print(f"Actual value    : {actual_value:.2f}")


if __name__ == "__main__":
    main()