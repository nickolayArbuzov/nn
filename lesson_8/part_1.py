import numpy as np
from keras.datasets import mnist


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


train_images_count = 1000
test_images_count = 10000
pixels_per_image = 28 * 28
digits_num = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_images = (
    x_train[:train_images_count].reshape(train_images_count, pixels_per_image) / 255.0
)
test_images = (
    x_test[:test_images_count].reshape(test_images_count, pixels_per_image) / 255.0
)


def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


train_labels = one_hot_encode(y_train[:train_images_count], digits_num)
test_labels = one_hot_encode(y_test[:test_images_count], digits_num)

np.random.seed(42)
hidden_size = 100

weights_in_hidden_1 = np.random.randn(pixels_per_image, hidden_size) * np.sqrt(
    2.0 / (pixels_per_image + hidden_size)
)
weights_hidden_1_out = np.random.randn(hidden_size, digits_num) * np.sqrt(
    2.0 / (hidden_size + digits_num)
)

learning_rate = 0.01
num_epochs = 100

for epoch in range(num_epochs):
    correct_answers = 0
    for j in range(len(train_images)):
        layer_in = train_images[j : j + 1]
        layer_hidden_1 = relu(np.dot(layer_in, weights_in_hidden_1))
        dropout_mask = np.random.randint(2, size=layer_hidden_1.shape)
        layer_hidden_1 *= dropout_mask * 2
        layer_out = softmax(np.dot(layer_hidden_1, weights_hidden_1_out))

        correct_answers += int(
            np.argmax(layer_out) == np.argmax(train_labels[j : j + 1])
        )

        layer_out_delta = layer_out - train_labels[j : j + 1]
        layer_hidden_1_delta = np.dot(
            layer_out_delta, weights_hidden_1_out.T
        ) * relu_derivative(layer_hidden_1)

        weights_hidden_1_out -= learning_rate * np.dot(
            layer_hidden_1.T, layer_out_delta
        )
        weights_in_hidden_1 -= learning_rate * np.dot(layer_in.T, layer_hidden_1_delta)

    print(
        f"Epoch {epoch + 1}: Training Accuracy = {correct_answers * 100 / len(train_images):.2f}%"
    )

correct_answers = 0
for i in range(len(test_images)):
    layer_in = test_images[i : i + 1]
    layer_hidden_1 = relu(np.dot(layer_in, weights_in_hidden_1))
    layer_out = softmax(np.dot(layer_hidden_1, weights_hidden_1_out))
    correct_answers += int(np.argmax(layer_out) == np.argmax(test_labels[i : i + 1]))

print(f"Test Accuracy: {correct_answers * 100 / len(test_images):.2f}%")
