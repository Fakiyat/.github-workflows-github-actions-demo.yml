import numpy as np

def bipolar_step_function(x):
    return np.where(x >= 0, 1, -1)

def and_gate_perceptron(inputs):
    weights = np.array([1, 1, 1, 1, 1])  # Weights for the five inputs
    bias = -4  # Bias

    weighted_sum = np.dot(inputs, weights) + bias
    output = bipolar_step_function(weighted_sum)

    return output

# Test the perceptron with different inputs
input_combinations = [np.array([-1, -1, -1, -1, -1]),
                       np.array([-1, -1, -1, -1, 1]),
                       np.array([-1, -1, -1, 1, 1]),
                       np.array([1, 1, 1, 1, 1])]

for input_combination in input_combinations:
    result = and_gate_perceptron(input_combination)
    print(f"Inputs: {input_combination}, Output: {result}")
