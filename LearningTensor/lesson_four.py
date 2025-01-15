
"""Basic classification: Classify images of clothing"""
import matplotlib.pylab
import tensorflow
import tf_keras
import numpy
import matplotlib.pyplot

fashion_data = tensorflow.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_data.load_data()

given_labels = train_labels
print(given_labels) # NOTE use the max() to find the max index and plus one to find the total number of items

#you can try to find all the classes by printing them out before storing the grousp into an array
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""Explore the data"""
print(f"This tells me (number of images, each picture is # length, each picture is # width) for the train images in the dataset: {train_images.shape}")
print(f"The number of photos in the training dataset: {len(train_labels)}")
print(f"The labels that are represented by an index value {train_labels}")
print(f"This tells me (number of images, each picture is # length, each picture is # width) for the test images for the dataset {test_images.shape}")
print(f"The number of photos in the testing dataset: {len(test_labels)}")

"""Preprocess the data"""
matplotlib.pyplot.figure()
matplotlib.pyplot.imshow(train_images[1])
matplotlib.pyplot.colorbar()
matplotlib.pyplot.grid(False)
matplotlib.pyplot.show()

# trainign set and the testing set be preprocessed in the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

# build and train the network
matplotlib.pyplot.figure(figsize = (10,10))
for i in range(25):
    matplotlib.pyplot.subplot(5,5,i+1)
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.yticks([])
    matplotlib.pyplot.grid(False)
    matplotlib.pyplot.imshow(train_images[i], cmap = matplotlib.pyplot.cm.binary)
    matplotlib.pyplot.xlabel(class_names[train_labels[i]])
matplotlib.pyplot.show()

"""Build and train your model (even though this can be put into a class we are putting it in a regular variable)"""
# new_model = tf_keras.src.engine.sequential.Sequential(
#                                     [tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
#                                          tensorflow.keras.layers.Dense(128, activation = 'relu'),
#                                          tensorflow.keras.layers.Dense(20)] # this is the final layer
# )

# the model being made
new_model = tensorflow.keras.Sequential([tensorflow.keras.layers.Flatten(), # input_shape=(28, 28)
                                         tensorflow.keras.layers.Dense(128, activation = 'relu'),
                                         tensorflow.keras.layers.Dense(20)]) # NOTE the first layer in the network

# compile the model
new_model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model
new_model.fit(train_images, train_labels, epochs=10)

# evaluate the accuracy
test_loss, test_accuracy = new_model.evaluate(test_images, test_labels, verbose = 2)
print(f"Test accuracy: {test_accuracy}")

"""Create a conclusion"""
prob_model = tensorflow.keras.Sequential([new_model, tensorflow.keras.layers.Softmax()])
overall_predictions = prob_model.predict(test_images)

#NOTE to be a picutre in the dataset to see its info
print(overall_predictions[0])
# the highest confidence value (from the 10 images of cothing)
# (the model believes what the image is mostly identify as)
print(numpy.argmax(overall_predictions[0]))
# the index number the label is refernce to 
print(test_labels[0])

# methods to graph the full set of 10 class predictions
def plot_image(i, predict_array, true_label, img):
    true_label, img = true_label[i], img[i]
    matplotlib.pyplot.grid(False)
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.yticks([])

    matplotlib.pyplot.imshow(img, cmap = matplotlib.pyplot.cm.binary)

    predicted_label = numpy.argmax(predict_array)
    if predicted_label == true_label: # matches
        color = 'blue'
    else: 
        color = 'red'

    matplotlib.pyplot.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 
                                       100*numpy.max(overall_predictions), 
                                       class_names[true_label],
                                       color = color))

def plot_value_array(i, predict_array, true_label):
    true_label = true_label[i]
    matplotlib.pyplot.grid(False)
    
    matplotlib.pyplot.xticks(range(len(predict_array)), [str(x) for x in range(len(predict_array))], rotation = 35)
    matplotlib.pyplot.yticks([])
    evaluation_plot = matplotlib.pyplot.bar(range(len(predict_array)),
                                            predict_array,
                                            color = "#777777")
    matplotlib.pyplot.ylim([0,1])
    predict_label = numpy.argmax(predict_array)

    evaluation_plot[predict_label].set_color('red')
    evaluation_plot[true_label].set_color('blue')

i = 0
matplotlib.pyplot.figure(figsize=(6,3))
plot_image(i, overall_predictions[i], test_labels, test_images)
matplotlib.pyplot.subplot(2,3,6)
plot_value_array(i, overall_predictions[i], test_labels)
matplotlib.pyplot.show()

num_rows = 5
num_cols = 3

num_images = num_rows*num_cols
matplotlib.pyplot.figure(figsize = (2*2*num_cols, 2*num_rows))
for i in range(num_images):
    matplotlib.pyplot.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, overall_predictions[i], test_labels, test_images)
    matplotlib.pyplot.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, overall_predictions[i], test_labels)
    # matplotlib.pyplot.show() NOTE for individual plots
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()

#train the model since you are able to display the model...
# the trianing will be on a singlel image to test its cofindence
first_img = test_images[0]
img_batch_made = (numpy.expand_dims(first_img, 0)) # one image is added into the batch
predict_single = prob_model.predict(img_batch_made)
plot_value_array(1, overall_predictions[0], test_labels)
_ = matplotlib.pyplot.xticks(range(10), class_names, rotation = 45) # used as a throwaway variable that is goign to be used later
matplotlib.pyplot.show()
