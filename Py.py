import numpy as np
# Activation functions and their derivatives
def relu(z):
    return np.maximum(0.0, z)


def relu_grad_mask(z):
    """d/dz relu(z), elementwise: 1 where z>0, else 0."""
    return (z > 0).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    # subtract row-max for numerical stability; does not change the result
    # since softmax is invariant to adding a constant to every logit.
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# The network
class TwoLayerNet:
    """
    A single-hidden-layer feedforward network:
        Linear -> ReLU -> Linear -> Softmax (multiclass head)
    Parameters are stored in a plain dict so they can be looped over
    generically by the optimizer and the gradient checker.
    """

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        # He initialization for the ReLU layer, Xavier-ish for the output layer.
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden)),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, np.sqrt(1.0 / n_hidden), size=(n_hidden, n_out)),
            "b2": np.zeros(n_out),
        }
        self.cache = {}

    # ----- forward -----
    def forward(self, X):
        W1, b1, W2, b2 = (self.params[k] for k in ("W1", "b1", "W2", "b2"))
        Z1 = X @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2

    @staticmethod
    def cross_entropy_loss(A2, Y_onehot):
        N = A2.shape[0]
        eps = 1e-12  # avoid log(0)
        return -np.sum(Y_onehot * np.log(A2 + eps)) / N

    # ----- backward (manual, see derivation #1 above) -----
    def backward(self, Y_onehot):
        X, Z1, A1, A2 = (self.cache[k] for k in ("X", "Z1", "A1", "A2"))
        W2 = self.params["W2"]
        N = X.shape[0]

        dZ2 = (A2 - Y_onehot) / N                # (N, n_out)
        dW2 = A1.T @ dZ2                         # (H, n_out)
        db2 = np.sum(dZ2, axis=0)                # (n_out,)

        dA1 = dZ2 @ W2.T                         # (N, H)
        dZ1 = dA1 * relu_grad_mask(Z1)            # (N, H)
        dW1 = X.T @ dZ1                          # (n_in, H)
        db1 = np.sum(dZ1, axis=0)                # (H,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


class TwoLayerNetBinary:
    """
    Stretch-goal activation/loss combo:
        Linear -> Tanh -> Linear -> Sigmoid -> Binary Cross-Entropy
    (see derivation #2 in the module docstring)
    """

    def __init__(self, n_in, n_hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(1.0 / n_in), size=(n_in, n_hidden)),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, np.sqrt(1.0 / n_hidden), size=(n_hidden, 1)),
            "b2": np.zeros(1),
        }
        self.cache = {}

    def forward(self, X):
        W1, b1, W2, b2 = (self.params[k] for k in ("W1", "b1", "W2", "b2"))
        Z1 = X @ W1 + b1
        A1 = tanh(Z1)
        Z2 = A1 @ W2 + b2
        A2 = sigmoid(Z2)
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2

    @staticmethod
    def bce_loss(A2, Y):
        N = A2.shape[0]
        eps = 1e-12
        return -np.sum(Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps)) / N

    def backward(self, Y):
        X, Z1, A1, A2 = (self.cache[k] for k in ("X", "Z1", "A1", "A2"))
        W2 = self.params["W2"]
        N = X.shape[0]

        dZ2 = (A2 - Y) / N                       # (N, 1)
        dW2 = A1.T @ dZ2                         # (H, 1)
        db2 = np.sum(dZ2, axis=0)                # (1,)

        dA1 = dZ2 @ W2.T                         # (N, H)
        dZ1 = dA1 * (1 - A1 ** 2)                # tanh'(z) = 1 - tanh(z)^2
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}



# Numerical gradient checking (central finite differences)

def numerical_gradient_check(net, X, Y, loss_fn, forward_fn=None, eps=1e-5,
                              n_checks_per_param=8, seed=0):
    """
    For each parameter tensor, randomly sample a handful of entries and
    compare the analytic gradient (from net.backward) against the
    central-difference numerical gradient:

        dL/dtheta_i ~= [ L(theta_i + eps) - L(theta_i - eps) ] / (2*eps)

    Returns a dict: {param_name: max_relative_error}
    """
    forward_fn = forward_fn or net.forward
    rng = np.random.default_rng(seed)

    A2 = forward_fn(X)
    analytic_grads = net.backward(Y)

    results = {}
    for name, param in net.params.items():
        flat = param.reshape(-1)
        analytic_flat = analytic_grads[name].reshape(-1)
        idxs = rng.choice(flat.size, size=min(n_checks_per_param, flat.size),
                           replace=False)
        max_rel_err = 0.0
        for i in idxs:
            orig = flat[i]

            flat[i] = orig + eps
            loss_plus = loss_fn(forward_fn(X), Y)

            flat[i] = orig - eps
            loss_minus = loss_fn(forward_fn(X), Y)

            flat[i] = orig  # restore
            numeric_grad = (loss_plus - loss_minus) / (2 * eps)
            analytic_grad = analytic_flat[i]

            denom = max(abs(numeric_grad), abs(analytic_grad), 1e-8)
            rel_err = abs(numeric_grad - analytic_grad) / denom
            max_rel_err = max(max_rel_err, rel_err)
        results[name] = max_rel_err
    return results


# Optimizers
class SGD:
    def __init__(self, params, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


class Adam:
    """Stretch goal: Adam optimizer, implemented manually (Kingma & Ba, 2015)."""

    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)