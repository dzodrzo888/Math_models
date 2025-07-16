import numpy as np

class NN:
    """
    This class is used to create a simple neural network.
    """
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = []
        self.biases = []

    def compute_predictions(self, X: np.array, w: float, b: float) -> np.array:
        """
        Calculates predictions.

        Args:
            x (np.array): Input values.
            w (float): Weights.
            b (float): Bias.

        Returns:
            z(np.array): Prediction
        """
        z = np.dot(X, w) + b

        return z
    
    def softmax_calc(self, z: np.array) -> np.array:
        """
        Function to do softmax calculation

        Args:
            z (np.array): Logit

        Returns:
            a (np.array): 
        """
        ez = np.exp(z)

        a = ez/np.sum(ez)

        return a

    def sigmoid_calc(self, z: np.array) -> np.array:
        """
        Function to calculate sigmoid value

        Args:
            z (np.array): Logit
        
        Returns:
            a: sigmoid
        """
        a = 1/(1 + np.exp(-z))

        return a
    
    def relu_calc(self, z: np.array) -> np.array:
        """
        Relu activation calculation

        Args:
            z (np.array): Logit

        Returns:
            float: 
        """
        return np.maximum(0,z)
    
    def _activation(self, z: np.array, kind="binary_crossentropy") -> np.array:
        """
        Function to apply a activation method

        Args:
            z (np.array): Logit
            kind (str): kind of activation function to be used.

        Raises:
            ValueError: Raised if invalid activation function kind inputed.

        Returns:
            np.array: Activation result
        """

        if kind == "softmax":
            return self.softmax_calc(z=z)
        elif kind == "relu":
            return self.relu_calc(z=z)
        elif kind == "sigmoid":
            return self.sigmoid_calc(z=z)
        else:
            raise ValueError("Invalid activation function inputed. Supported function: ['softmax', 'relu', 'sigmoid']")
        
    def _activation_derivative(self, z: float, kind="binary_crossentropy"):
        """
        Calculates the derivatives of the activation function

        Args:
            z (float): Activation function result
            kind (str): Type of activation function

        Raises:
            ValueError: Raised if invalid activation function kind inputed.

        Returns:
            float/int: Derivative of the activation
        """
        if kind == "sigmoid":
            return z * (1 - z)
        elif kind == "relu":
            return np.where(z > 0, 1, 0)
        else:
            raise ValueError("Invalid activation function inputed. Supported function: ['relu', 'sigmoid']")
        
    def _MSE_calc(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculates mean squared error

        Args:
            y_true (np.array): Y true value
            y_pred (np.array): Y predicted value

        Returns:
            mse (float): Mean squared error
        """
        mse = np.mean(np.square(y_true - y_pred))
        return mse
    
    def _binary_crossentropy_calc(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculates binary crossentropy

        Args:
            y_true (np.array): Y true value.
            y_pred (np.array): Y predicted value.

        Returns:
            bin_loss_mean (float): Binary crossentropy loss value.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        bin_loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

        bin_loss_mean = -np.mean(bin_loss)

        return bin_loss_mean

    def _compute_loss(self, y_true, y_pred, kind="binary_crossentropy"):
        """
        Computes loss based on a function

        Args:
            y_true (np.array): Y true value.
            y_pred (np.array): Y predicted value.
            kind (str): Kind of loss function

        Raises:
            ValueError: Raised if invalid activation function kind inputed.

        Returns:
            np.array: Array of losses
        """
        if kind == "mean_squared_error":
            return self._MSE_calc(y_true=y_true, y_pred=y_pred)
        elif kind == "binary_crossentropy":
            return self._binary_crossentropy_calc(y_true=y_true, y_pred=y_pred)
        else:
            raise ValueError("Invalid loss function inputed. Supported function: ['mean_squared_error', 'binary_crossentropy']")
        
    
    def _dense(self, A_in: np.array, W: np.array, b: np.array, kind="binary_crossentropy") -> np.array:
        """
        NN layer used to 

        Args:
            A_in (np.array): Input vector
            W (np.array): Weights vector
            b (np.array): Bias vector

        Returns:
            A_out(np.array): Output activation after applying activation function.
        """
        Z = np.matmul(A_in, W) + b

        A_out = self._activation(Z, kind=kind)

        return A_out, Z

    def _forward_prop(self, X: np.array) -> np.array:
        """
        Forward propagation func used to calculate the propability

        Args:
            X (np.array): Input vector

        Returns:
            A_s(np.array): Output activation after applying activation function.
            Z_s(np.array): Linear transformation before activation.
        """
        A = X
        A_s = [A]
        Z_s = []

        for i in range(len(self.weights)):
            A, Z = self._dense(A, self.weights[i], self.biases[i], kind=self.activations[i])
            A_s.append(A)
            Z_s.append(Z)

        return A_s, Z_s 

    def _backward(self, A_s: np.array, y_true: np.array):
        """
        Function used to compute gradients using the backward propagation

        Args:
            A_s(np.array): Output activation after applying activation function.
            Z_s(np.array): Linear transformation before activation.
            y_true (np.array): Y true value

        Returns:
            grads_w(list[float]): Gradients of weights.
            grads_b(list[float]): Gradients of biases.
        """
        grads_w = []
        grads_b = []

        A_last = A_s[-1]
        delta = A_last - y_true 

        for i in reversed(range(len(self.weights))):

            dw = np.dot(A_s[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i > 0:

                d_activation = self._activation_derivative(A_s[i], kind=self.activations[i])
                delta = np.dot(delta, self.weights[i].T) * d_activation

        return grads_w, grads_b

    def _update_parameters(self, grads_w: list[float], grads_b: list[float]):
        """
        Updates weights based on the learning rate

        Args:
            grads_w(list[float]): Gradients of weights.
            grads_b(list[float]): Gradients of biases.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def fit(self, X: np.array, y: np.array, hidden_layers=[5], activations=["relu", "sigmoid"], loss="binary_crossentropy"):
        """
        Fit the neural network model.

        Args:
            X (np.array): Input features.
            y (np.array): Target labels.
            hidden_layers (list[int]): Number of units in each hidden layer.
            activations (list[str]): Activation function per layer.
            loss (str): Loss function to use.
        """
        np.random.seed(42)
        input_dim = X.shape[1]
        layer_dims = [input_dim] + hidden_layers + [1]

        self.activations = activations

        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i + 1]) * 0.01
            b = np.zeros((1, layer_dims[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        for epoch in range(self.epochs):
            A_s, Z_s = self._forward_prop(X)
            loss_val = self._compute_loss(y, A_s[-1], kind=loss)
            grads_w, grads_b = self._backward(A_s, Z_s, y)
            self._update_parameters(grads_w, grads_b)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} | Loss: {loss_val:.4f}")

    def predict(self, X, threshold=0.5):
        """
        Predicts binary values based on propability

        Args:
            X (_type_): _description_
            threshold (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """
        cls_labels = (self.predict_proba(X) >= threshold).astype(int)
        
        return cls_labels

    def predict_proba(self, X):
        A_s, _ = self._forward_prop(X=X)

        y_hat_prob = A_s[-1]

        return y_hat_prob

if __name__ == "__main__":
    X = np.random.randn(100, 2)
    y = (np.random.rand(100, 1) > 0.5).astype(float)

    model = NN(learning_rate=0.001, epochs=100)
    model.fit(X, y, hidden_layers=[4], activations=["relu", "sigmoid"], loss="binary_crossentropy")
    print(model.predict(X=X))
