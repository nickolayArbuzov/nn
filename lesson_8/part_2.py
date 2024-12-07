import numpy as np
from keras.datasets import mnist


def relu(x):
    return (x > 0) * x


def reluderiv(x):
    return x > 0


train_images_count = 1000
test_images_count = 10000
pixels_per_image = 28 * 28
digits_num = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = (
    x_train[0:train_images_count].reshape(train_images_count, pixels_per_image) / 255
)
train_labels = y_train[0:train_images_count]

test_images = (x_test)[0:test_images_count].reshape(
    test_images_count, pixels_per_image
) / 255
test_labels = y_test[0:test_images_count]

one_hot_labels = np.zeros((len(train_labels), digits_num))
for i in range(len(train_labels)):
    one_hot_labels[i][train_labels[i]] = 1

train_lables = one_hot_labels

one_hot_labels = np.zeros((len(test_labels), digits_num))
for i, j in enumerate(test_labels):
    one_hot_labels[i][j] = 1

test_labels = one_hot_labels

np.random.seed(2)
hidden_size = 50

weights_in_hidden_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_hidden_1_out = 0.2 * np.random.random((hidden_size, digits_num)) - 0.1

learning_rate = 0.01
num_epochs = 100
batch_size = 50

for i in range(num_epochs):
    correct_answers = 0
    for j in range(int(len(train_images) / batch_size)):
        batch_start = batch_size * j
        batch_end = batch_size * (j + 1)
        layer_in = train_images[batch_start:batch_end]
        layer_hidden_1 = relu(np.dot(layer_in, weights_in_hidden_1))
        dropout_mask = np.random.randint(2, size=layer_hidden_1.shape)
        layer_hidden_1 *= dropout_mask * 2
        layer_out = np.dot(layer_hidden_1, weights_hidden_1_out)
        for k in range(batch_size):
            correct_answers += int(
                np.argmax(layer_out[k : k + 1])
                == np.argmax(train_labels[batch_start + k : batch_start + k + 1])
            )

        layer_out_delta = (layer_out - train_labels[batch_start:batch_end]) / batch_size
        layer_hidden_1_delta = layer_out_delta.dot(weights_hidden_1_out.T) * reluderiv(
            layer_hidden_1
        )

        weights_hidden_1_out -= learning_rate * layer_hidden_1.T.dot(layer_out_delta)
        weights_in_hidden_1 -= learning_rate * layer_in.T.dot(layer_hidden_1_delta)
    print("Epoch: ", i)
    print("Accuracy: ", correct_answers * 100 / len(train_images))

correct_answers = 0
for i in range(len(test_images)):
    layer_in = test_images[i : i + 1]
    layer_hidden_1 = relu(np.dot(layer_in, weights_in_hidden_1))
    layer_out = np.dot(layer_hidden_1, weights_hidden_1_out)
    correct_answers += int(np.argmax(layer_out) == np.argmax(test_labels[i : i + 1]))

print("Test Accuracy: ", correct_answers * 100 / len(test_images))
