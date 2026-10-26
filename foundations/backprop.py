import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        def sigmoid(z):
            return np.round(1/(1 + np.exp(-z)), 5)
        def sigmoid_prime(z):
            sig = sigmoid(z)
            return np.round(sig*(1-sig), 5)

        n = len(x)
        
        z = np.dot(x, w) + b
        y_hat = sigmoid(z)

        dL_dw = [0] * n

        for i in range(n):
            dL_dw[i] = np.round((y_hat-y_true)*sigmoid_prime(z)*x[i], 5)
        
        dL_db = np.round((y_hat-y_true)*sigmoid_prime(z), 5)

        return np.array(dL_dw), dL_db