import numpy as np
import matplotlib.pyplot as plt


# Monotonic ascending functions (non-negative)
def linear(t):
    return t

def logarithmic(t):
    return np.log1p(t)  # log(1 + t), starts at 0

def cubic(t):
    return (t / 100) ** 3  # scaled cubic to stay in [0, 1]

def sigmoid(t):
    return 1 / (1 + np.exp(-(t - 50) / 10))  # range (0, 1)


def transform_function(func, t, A, B, C):
    """
    Apply amplitude scaling (A), vertical shift (B), and time shift (C) to a monotonic function.
    Ensures the output is 0 before time C, and truncated after time 100.

    Parameters:
        func (callable): Base monotonic function that takes an array t.
        t (np.ndarray): Time array in [0, 100).
        A (float): Amplitude scaling factor (must be > 0).
        B (float): Amplitude translation (must be > 0).
        C (float): Time shift (must be > 0).

    Returns:
        t (np.ndarray): The original time array.
        y_final (np.ndarray): The transformed function with zero padding before t = C.
    """
    if A < 0 or B < 0 or C < 0:
        raise ValueError("A, B, and C must be strictly positive.")

    # Create output array initialized to 0
    y_final = np.zeros_like(t)

    # Only apply the function where t >= C
    mask = t >= C
    t_shifted = t[mask] - C

    # Evaluate and transform the base function
    y_transformed = A * func(t_shifted) + B

    # Place the transformed values into the output
    y_final[mask] = y_transformed

    return t, y_final



def plot_monotonic_functions(t, y_linear, y_log, y_cubic, y_sigmoid):
    """
    Plots four monotonic ascending functions over the input t.

    Parameters:
        t (array-like): The x-axis values.
        y_linear (array-like): Linear function values.
        y_log (array-like): Logarithmic function values.
        y_cubic (array-like): Cubic function values.
        y_sigmoid (array-like): Sigmoid function values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(t, y_linear, label='Linear', linewidth=2)
    plt.plot(t, y_log, label='Logarithmic', linewidth=2)
    plt.plot(t, y_cubic, label='Cubic (normalized)', linewidth=2)
    plt.plot(t, y_sigmoid, label='Sigmoid-like', linewidth=2)
    plt.title('4 Monotonic Ascending Functions (Non-negative)')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# === Function to sum all transformed functions ===
def sum_transformed_functions(t, functions, transform_params):
    """
    Sums the transformed functions from the provided dictionary.
    
    Parameters:
      t (np.ndarray): The time vector over [0, 100).
      functions (dict): Dictionary mapping names to base functions.
      transform_params (dict): Dictionary mapping function names to (A, B, C) tuples.
      
    Returns:
      t (np.ndarray): The original time vector.
      y_sum (np.ndarray): The pointwise sum of all transformed function outputs.
    """
    y_sum = np.zeros_like(t)
    for label, func in functions.items():
        A, B, C = transform_params[label]
        _, y_trans = transform_function(func, t, A, B, C)
        y_sum += y_trans
    return t, y_sum
