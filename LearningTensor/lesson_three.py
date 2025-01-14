"""
TensorFlow part 2 quick start for experts

Steps:
1) In Colab, connect to a python runtime: At the top-right of 
the menu bar, select CONNECT.
2) Run all the notebook code cells: Select Runtime > Run all.

"""

import tensorflow
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#NOTE load the MNIST dataset

chosen_dataset = tensorflow.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = chosen_dataset.load_data()

# channels dimension ***each channel represents a different color components of each pixel***
#NOTE this is for grayscale, the color is x_train.astype("float32")
x_train = x_train[..., tensorflow.newaxis].astype("float32")
x_test = x_test[..., tensorflow.newaxis].astype("float32")

# shuffle the dataset and batch (batch means to divide datasets into small, manageable groups of data points for efficient training)
train_dataset = tensorflow.data.Dataset.from_tensor_slices(
                                                            (x_train, y_train)).shuffle(10000).batch(32)
test_dataset = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

"""NOTE: BUILD A MODEL using Keras
its alot easier than rewriting code like this a jillion times
    model = tf_keras.src.engine.sequential.Sequential(
                                    [tensorflow.keras.layers.Flatten(input_shape = (28,28)),
                                     tensorflow.keras.layers.Dense(128, activation = 'relu'),
                                     tensorflow.keras.layers.Dropout(0.2),
                                     tensorflow.keras.layers.Dense(10)] # this is the final layer
)
"""
class MyModel(Model):
    def __init__(self):
        self.conv = Conv2D(32, 3, activation = 'relu')
        self.flatten = Flatten()
        self.dense_one = Dense(128, activation = 'relu')
        self.dense_two = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense_one(x)
        return self.dense_two(x)

# to see if the class works
model = MyModel()

loss_obj = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tensorflow.keras.optimizers.Adam()

"""
predictions = model(x_train[:1]).numpy()
print(f"This is the prediction: {predictions}") # NOTE returns a vector of logits or log-odds scores, one for each class
probabilities = tensorflow.nn.softmax(predictions).numpy()
print(f"This is the probabilities: {probabilities}")#NOTE softmax = the probability of the raw scores
"""
prob_loss_training = tensorflow.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')


"""Create a method to train model"""
#NOTE @tf.function a decorator that transform a method into a tensor flow graph
#NOTE can be left inisde or outside of teh class
@tensorflow.function
def train_step(images, labels):
    with tensorflow.GradientTape() as tape:
        #NOTE f
        predict = model(images, training = True)
        loss = loss_object(labels, predict)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    prob_loss_training(loss)
    train_accuracy(labels, predict)

# test method
@tensorflow.function
def test_step(images, labels):
    # training = False is only needed if there are layers with different behaviors
    predict = model(images, training = False)
    test_loss = loss_obj(labels, predict)

    prob_loss_training(test_loss)
    train_accuracy(labels, predict)

EPOCHS = 5
for epoch in range(EPOCHS):
    # reset the metrics EPOCHS (training iteration)
    prob_loss_training.reset_state()
    train_accuracy.reset_state()
    #
    #