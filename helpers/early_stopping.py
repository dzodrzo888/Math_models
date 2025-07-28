"""This module is used to implement early stopping."""
import numpy as np

class EarlyStopping:

    def __init__(self, delta: float, patience: int, mode: str):
        self.delta = delta
        self.patience = patience
        self.mode = mode
        self.epochs_no_improve = 0
        self.best_epoch_weights = None
        self.stop_training = False

        if self.mode == "min":
            self.best_value = np.inf
        elif self.mode == "max":
            self.best_value = 0.
        else:
            raise ValueError("Wrong mode value! Only input 'min' or 'max'")

    def on_epoch_end(self, curr_epoch_val: float, weights: np.ndarray) -> bool:
        """
        Checks if algorithm keeps imporiving by counting cost values.

        Args:
            curr_epoch_val (float): Value of current epoch.
            weights (np.ndarray): Weights of a model

        Returns:
            bool: Value monitoring if patience was exceeded.
        """

        significant_improvement = False

        if self.mode == "min":
            if curr_epoch_val < (self.best_value - self.delta):
                significant_improvement = True

            if curr_epoch_val < self.best_value:
                self.best_value = curr_epoch_val
                self.best_epoch_weights = weights
        elif self.mode == "max":

            if curr_epoch_val > (self.best_value + self.delta):
                significant_improvement = True

            if curr_epoch_val > self.best_value:
                self.best_value = curr_epoch_val
                self.best_epoch_weights = weights

        if significant_improvement:
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve > self.patience:
            self.stop_training = True

        return self.stop_training

    def reset(self) -> None:
        """
        Resets the epoch class.
        """

        self.epochs_no_improve = 0
        self.best_epoch_weights = None
        self.stop_training = False
        self.best_value = np.inf if self.mode == "min" else -np.inf

    def get_best_weights(self) -> np.ndarray:
        """
        Returns best weights encountered.

        Returns:
            np.ndarray: Best model weights.
        """
        return self.best_epoch_weights
    
    def get_best_cost_value(self) -> float:
        """
        Returns the best cost value encountered.

        Returns:
            float: Best float value.
        """
        return self.best_value
