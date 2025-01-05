import tensorflow
import tf_keras
from tf_keras import layers
import tensorflow_model_optimization as tfmot

"""
Quick Start into TensorFlow
1) get prebuild dataset (an MNIST dataset)
2) build a neutral netowork machine learning
3) Train this neural netowork
4) Evaluate the accuracy of the model.
Website used: https://www.tensorflow.org/tutorials/quickstart/beginner
"""

# The loading of a dataset
"""
keywords used:
x_train = input variables in dataset
y_train = out variables in dataset

x_test = training data of the machine learning model
"""
# NOTE dividing by 255 makes the integer number into floating point numberss


# Step one: prebuild dataset
data = tensorflow.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Step two: build neutral model (should create an output)
model = tf_keras.src.engine.sequential.Sequential(
                                    [tensorflow.keras.layers.Flatten(input_shape = (28,28)),
                                     tensorflow.keras.layers.Dense(128, activation = 'relu'),
                                     tensorflow.keras.layers.Dropout(0.2),
                                     tensorflow.keras.layers.Dense(10)] # this is the final layer
) # NOTE 'Sequential' is a function/formula used to create a model

"""
logits/log-odds = un-normalized scores(raw scores) taht are produced by the final layer
"""
predictions = model(x_train[:1]).numpy()
print(f"This is the prediction: {predictions}") # NOTE returns a vector of logits or log-odds scores, one for each class
probabilities = tensorflow.nn.softmax(predictions).numpy()
print(f"This is the probabilities: {probabilities}")#NOTE softmax = the probability of the raw scores
"""
How its done:
softmax(zi) = exp(zi) / sum(exp(zj)) for all j 
where:

zi is the logit for class i
exp() is the exponential function (e^x)

example:
This is the prediction: [[ 0.1383641   0.13797253 -0.02670263 -0.4118031  -0.143114    0.50748676
  -0.14279084  0.14581402  0.4932806   0.4338226 ]]
softmax= (math.pow(math.e, 0.1383641)/ ( math.pow(math.e, 0.1383641)  + 
                                         math.pow(math.e, 0.13797253)  + 
                                         math.pow(math.e, -0.02670263) + 
                                         math.pow(math.e, -0.4118031) + 
                                         math.pow(math.e, -0.143114) + 
                                         math.pow(math.e, 0.50748676) +
                                         math.pow(math.e, -0.14279084) + 
                                         math.pow(math.e, 0.14581402)  + 
                                         math.pow(math.e, 0.4932806) + 
                                         math.pow(math.e, 0.4338226)))

The anwser is 0.09844821550963065 which is equal to 0.09844822

This is the entire probability: [[0.09844822 0.09840968 0.08346805 0.05679019 0.07429566 0.1424019
                                  0.07431968 0.09918439 0.14039323 0.13228904]]
   
This is the class probaility = softmax
"""

"""
finding probability loss in each class
"""
losses_found = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
losses_found(y_train[:1], predictions).numpy()
model.compile(optimizer = 'adam' , 
              loss=losses_found,
              metrics = ['accuracy'])


# Step three: train neutral model (should porduce an output)
model.fit(x_train, y_train, epochs=5) # use to minize the losses made
#NOTE epochs = rounds of cases made
"""
Output look something like this:
Epoch 1/5
1875/1875 [==============================] - 1s 578us/step - loss: 0.3011 - accuracy: 0.9126
Epoch 2/5
1875/1875 [==============================] - 1s 573us/step - loss: 0.1449 - accuracy: 0.9574
Epoch 3/5
1875/1875 [==============================] - 1s 556us/step - loss: 0.1081 - accuracy: 0.9676
Epoch 4/5
1875/1875 [==============================] - 1s 552us/step - loss: 0.0887 - accuracy: 0.9729
Epoch 5/5
1875/1875 [==============================] - 1s 563us/step - loss: 0.0756 - accuracy: 0.9764

"""
# this checks the perfomance of the moel
model.evaluate(x_test, y_test, verbose=2)
# Output: 313/313 - 0s - loss: 0.0783 - accuracy: 0.9753 - 186ms/epoch - 594us/step
#NOTE loss = loss of raw data in the model
#NOTE accuracy = accuracy of raw data in the model
#NOTE the rate given is the average time taken to complete one full trainign epoch
"""NOTE The image classifier is now trained to ~98% accuracy on this dataset."""

# Step four: evaulate the model (what was the result)
"""You trained a machine learning model using Keras API"""

"""leason two is talk about more about image data loading and CSV loading"""