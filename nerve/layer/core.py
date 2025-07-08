import numpy as np

class Layer:
    """
    Base class for all layers
    """
    def __init__(self):
        pass

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        raise NotImplementedError
 
