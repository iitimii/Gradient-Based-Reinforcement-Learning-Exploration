import numpy as np
from typing import List, Literal, Union

class NNetwork:
    def __init__(self, layers: List[int], activation: Literal["relu", "sigmoid"] = "relu", 
                 optimizer: Literal["sgd", "adam"] = "sgd", learning_rate: float = 0.01) -> None:
        self.parameters = {}
        self.A_cache = {}
        self.Z_cache = {}
        self.num_layers = len(layers)
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        assert self.num_layers > 1, "Network must have at least two layers"

        for l in range(1, self.num_layers):
            self.parameters[f'W{l}' ] = np.random.randn(layers[l], layers[l - 1]) * 0.01
            self.parameters[f'b{l}' ] = np.zeros((layers[l], 1))
            print(self.parameters[f'W{l}'].shape)

        if optimizer == "adam":
            self.v = {k: np.zeros_like(v) for k, v in self.parameters.items()}
            self.s = {k: np.zeros_like(v) for k, v in self.parameters.items()}
            self.t = 0

    def empty_cache(self) -> None:
        self.A_cache = {}
        self.Z_cache = {}

    def _relu(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A = np.maximum(0, Z)
        return A, Z

    def _sigmoid(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A = 1 / (1 + np.exp(-Z))
        return A, Z
    
    def _relu_back(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    def _sigmoid_back(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        s = 1 / (1 + np.exp(-Z))
        return dA * s * (1 - s)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.empty_cache()
        self.A_cache['A0'] = X

        for l in range(1, self.num_layers):
            Z = np.dot(self.parameters[f'W{l}'], self.A_cache[f'A{l-1}']) + self.parameters[f'b{l}']
            self.Z_cache[f'Z{l}'] = Z
            
            if l == self.num_layers-1:  # Output layer
                A, _ = self._sigmoid(Z)
            else:
                A, _ = self._relu(Z) if self.activation == "relu" else self._sigmoid(Z)
            
            self.A_cache[f'A{l}'] = A

        return self.A_cache[f'A{self.num_layers-1}']


    def backward(self,  reward) -> dict:
        grads = {}

        loss = reward 
        dA = 0 # which cancels out the entire computational graph
        dA = reward * np.random.randn(1) # let's just imagine
        
        for l in reversed(range(1, self.num_layers)):
            Z = self.Z_cache[f'Z{l}']
            if l == self.num_layers - 1:
                dZ = self._sigmoid_back(dA, Z)
            else:
                dZ = self._relu_back(dA, Z) if self.activation == "relu" else self._sigmoid_back(dA, Z)
            
            A_prev = self.A_cache[f'A{l-1}']
            grads[f'dW{l}'] = np.dot(dZ, A_prev.T)
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True)
            
            if l > 1:
                dA = np.dot(self.parameters[f'W{l}'].T, dZ)

        return grads


    def optimizer_step_up(self, grads: dict) -> None:
        if self.optimizer == "sgd":
            for l in range(1, self.num_layers):
                self.parameters[f'W{l}'] += self.learning_rate * grads[f'dW{l}']
                self.parameters[f'b{l}'] += self.learning_rate * grads[f'db{l}']

        elif self.optimizer == "adam":
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
            self.t += 1
            
            for l in range(1, self.num_layers):
                for param in ['W', 'b']:
                    self.v[f'{param}{l}'] = beta1 * self.v[f'{param}{l}'] + (1 - beta1) * grads[f'd{param}{l}']
                    self.s[f'{param}{l}'] = beta2 * self.s[f'{param}{l}'] + (1 - beta2) * (grads[f'd{param}{l}']**2)
                    
                    v_corrected = self.v[f'{param}{l}'] / (1 - beta1**self.t)
                    s_corrected = self.s[f'{param}{l}'] / (1 - beta2**self.t)
                    
                    self.parameters[f'{param}{l}'] += self.learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.forward(X) > 0.5).astype(int)