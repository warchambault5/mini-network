import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0],
])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(X, W1, b1, W2, b2):
    hidden = sigmoid(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    return hidden, output


def train_and_save(filename="xor.npz", epochs=10000, learning_rate=0.5):
    np.random.seed(42)

    W1 = np.random.randn(2, 3)
    b1 = np.zeros((1, 3))

    W2 = np.random.randn(3, 1)
    b2 = np.zeros((1, 1))

    for epoch in range(epochs):
        hidden, output = forward(X, W1, b1, W2, b2)

        error = output - y

        d_output = error * output * (1 - output)

        d_W2 = hidden.T @ d_output
        d_b2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = (d_output @ W2.T) * hidden * (1 - hidden)

        d_W1 = X.T @ d_hidden
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        W2 -= learning_rate * d_W2
        b2 -= learning_rate * d_b2

        W1 -= learning_rate * d_W1
        b1 -= learning_rate * d_b1

        if epoch % 1000 == 0:
            loss = np.mean(error ** 2)
            print(f"epoch {epoch}, loss: {loss:.6f}")

    np.savez(
        filename,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2
    )

    print(f"\nweights saved: {filename}")

    return W1, b1, W2, b2

def load_weights(filename="xor.npz"):
    weights = np.load(filename)

    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]

    weights.close()

    print(f"weights loaded: {filename}")

    return W1, b1, W2, b2


def predict(X, W1, b1, W2, b2):
    _, output = forward(X, W1, b1, W2, b2)
    return output

W1, b1, W2, b2 = train_and_save()

#W1, b1, W2, b2 = load_weights()

predictions = predict(X, W1, b1, W2, b2)

print("\npredictions:")
print(predictions)

print("\nrounded predictions:")
print(np.round(predictions))
