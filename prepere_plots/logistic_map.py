import numpy as np
import matplotlib.pyplot as plt


def logistic_map(r, x0, num_iterations):
    result = []
    x = x0
    for _ in range(num_iterations):
        result.append(x)
        x = r * x * (1 - x)

    return result

# Example usage
r_value = 100  # You can experiment with different values of r
initial_value = 0.5
iterations = 50

logistic_sequence = logistic_map(r_value, initial_value, iterations)
plt.plot(logistic_sequence)
plt.show()