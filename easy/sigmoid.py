import math


def sigmoid(z: float) -> float:
    """Compute the sigmoid of a given input z.
    Args:
        z: A float value.
    Returns:
        The sigmoid of z as a float.
    """
    value = 1 / (1 + math.exp(-z))

    return round(value * 10000) / 10000  # rounding to 4 decimal places


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate values from -10 to 10
    x = np.linspace(-10, 10, 400)
    y = [sigmoid(val) for val in x]

    # Plot the sigmoid function
    plt.plot(x, y)
    plt.title("Sigmoid Function")
    plt.xlabel("z")
    plt.ylabel("sigmoid(z)")
    plt.grid()
    plt.axhline(0, color="black", linewidth=0.5, ls="--")
    plt.axvline(0, color="black", linewidth=0.5, ls="--")
    plt.show()
