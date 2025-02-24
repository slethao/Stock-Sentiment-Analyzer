import tensorflow
import tf_keras

"""
TensorFlow is a framework with models and api's (keras)
that are used to build machine learning applications
for task such as image recognition, 
text classifcation through models. These applications are
used for training, evaluating, and displaying
results with accuracy through Tensors 
which are multidimensional arrays used to represent data. 

The purpose of TensorFlow is used to 
run neural networks which are the various
layers within a TensorFlow model that is used to learn patterns
found in the given dataset, make predictions based on 
learned relationships in the data, 
automate tasks and perform calculations for results 
more efficient.

Quick Start into TensorFlow:
NOTE We will be using a prebuild dataset MNIST that
is given to us by TensorFlow.
1) get prebuild dataset (an MNIST dataset)
2) build a neural network machine learning model
3) Train this neural network
4) Evaluate the performance of the model.
"""


"""
@TODO: Step one: Loading of a prebuild dataset

The MNIST dataset will be loaded into the program file.
Then formatted appropriately by spliting the dataset
into x_train, y_train, x_test, y_test *reference NOTE below*
to make it accessable for the machine learning model to 
process the data.

NOTE the MNIST dataset contains images of handwritten digits 
from 0 through 9 that have pixel values that range from 0 to 255.

NOTE keywords used:
x_train = input variables in dataset
y_train = output variables in dataset
x_test = testing data of the machine learning model
y_test = testing data of the machine learning model

NOTE dividing by 255 makes the integer number into floating point numbers
****Programmed example of how its done is below****
"""
data = tensorflow.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


"""
@TODO: Step two: build neural network model (should create an output) 

After the MNIST dataset is loaded into the program, the programmer must
build a neural network model that will be used to train the dataset to 
come up with a metrics such as accuracy to see how effectively it can 
classifies the handwritten digits given. In addition, the programmer will 
also use loss function to determine how well the model is performing during 
training which will later show patterns between pixels values in the 
handwritten digits given.

Terminology used:
  Flatten: This layer converts the 28x28 pixel images into a 1D array.
  Dense: These are fully connected layers. The 'relu' activation function introduces non-linearity.
  softmax: The final layer, which outputs probabilities for each digit (0-9).
  logits/log-odds = un-normalized scores(raw scores) that are produced by the final layer

The formula for softmax:
  softmax(zi) = exp(zi) / sum(exp(zj)) for all j 
  Variables used:
    zi is the logit for class i
    exp() is the exponential function (e^x)

  Example:
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

      NOTE This is the class probaility = softmax(logits)
****Programmed example of how its done is below****
"""
model = tf_keras.src.engine.sequential.Sequential(
                                    [tensorflow.keras.layers.Flatten(input_shape = (28,28)),
                                     tensorflow.keras.layers.Dense(128, activation = 'relu'),
                                     tensorflow.keras.layers.Dropout(0.2),
                                     tensorflow.keras.layers.Dense(10)] # this is the final layer
) # NOTE 'Sequential' is a function/formula used to create a model

predictions = model(x_train[:1]).numpy()
print(f"This is the prediction: {predictions}") # NOTE returns a vector of logits or log-odds scores, one for each class
probabilities = tensorflow.nn.softmax(predictions).numpy()
print(f"This is the probabilities: {probabilities}")#NOTE softmax = the probability of the raw scores

# finding probability loss in each class
losses_found = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
losses_found(y_train[:1], predictions).numpy()
model.compile(optimizer = 'adam' , 
              loss=losses_found,
              metrics = ['accuracy'])

"""
@TODO: Step three: train neural network model (should porduce an output)
explaination here and content
Output look something like this: (shows the perfomance of the moel)
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


NOTE epochs = rounds of cases made
****Programmed example of how its done is below****
"""
model.fit(x_train, y_train, epochs=5) # use to minize the losses made


"""
@TODO: Step four: evaulate the model (what was the result)
Output factors displayed:
    Line displayed in the Terminal: 313/313 - 0s - loss: 0.0783 - accuracy: 0.9753 - 186ms/epoch - 594us/step
loss = loss of raw data in the model
accuracy = accuracy of raw data in the model

The image classifier is now trained to ~98% accuracy on this dataset.
(conte4nt and explaination)

You trained a machine learning model using Keras API

NOTE the rate given is the average time taken to complete one full trainign epoch
****Programmed example of how its done is below****
"""
model.evaluate(x_test, y_test, verbose=2)
