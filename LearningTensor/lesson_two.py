"""leason two is talk about more about image data loading and CSV loading"""

#Step 1: Set up
import matplotlib.axis
import matplotlib.figure
import numpy
import os
import PIL
import glob
import PIL.Image
import tensorflow
from tensorflow import keras
import matplotlib.pyplot
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import pathlib

"""The example we will be using is the dataset on photos of flowers"""
#Step 2: download and create a folder for the content that will be used
given_set = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tensorflow.keras.utils.get_file(
                                           'flower_photos', 
                                           origin=given_set, 
                                          untar = True)
data_dir = pathlib.Path("LearningTensor/flower_photos")
image_count = (len(list(data_dir.glob('*/*.jpg')))) 
print(f"image count: {image_count}") 

#Step 3: Display the image you want to see
"""Look at each directory"""
roses = list(data_dir.glob('roses/*.jpg'))
display_rose = PIL.Image.open(str(roses[1]))
display_rose.show()

"""Load the data with Keras"""
#Step 1: Define the parameters for the image
number_of_products = 32
img_height = 180
img_width = 180

#Step 2: using a validation split when training a model
model_training = tensorflow.keras.utils.image_dataset_from_directory(
                                                                       data_dir,
                                                                       validation_split = 0.2, # this is what I'm refering to (using 20% of the files to do validation but the 80% would be use for training)
                                                                       subset = "training", #NOTE switches the mode of whether or not you want 'training' or 'validation' of dataset
                                                                       seed = 123,
                                                                       image_size = (img_height, img_width),
                                                                       batch_size = number_of_products
)
print(10*"-")
valid_model_training = tensorflow.keras.utils.image_dataset_from_directory(
                                                                            data_dir,
                                                                            validation_split = 0.2, #NOTE 20% is for training and the 80% is for validation
                                                                            subset = "validation",
                                                                            seed = 123,
                                                                            image_size = (img_height, img_width),
                                                                            batch_size = number_of_products
)
classes_used = model_training.class_names
print(classes_used)

matplotlib.pyplot.figure(figsize = (10, 10))
for images, labels in model_training.take(1):
    for i in range(9): # if you want 9 images you need to subtract that number by 1
        ax = matplotlib.pyplot.subplot(3,3, i+1)
        matplotlib.pyplot.imshow(images[i].numpy().astype("uint8"))
        matplotlib.pyplot.title(classes_used[labels[i]])
        matplotlib.pyplot.axis("off")
    matplotlib.pyplot.show()

"""
Manually iterate over the dataset and retrieve batches of images
(this is manual but not recomemended to use the break in this way)
"""
for img_batch, label_batch in model_training:
    print(f"{img_batch.shape} This is the shape of the image")
    print(f"{label_batch.shape} This is the number that represents the batch of the images of shape")
    break

"""
Here, you will standardize values to be in the [0, 1] range by using
tf.keras.layers.Rescaling:
normalization layer is a technique used to stabilize the training process and improve the performance of neural networks.
"""
normalization_layer = tensorflow.keras.layers.Rescaling(1./255)

"""
How to use the normalize layer
you can use it on a dataset map
"""

norm_dataset = model_training.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(norm_dataset))
first_image = image_batch[0]
# pixel values
print(numpy.min(first_image), numpy.max(first_image))

"""lesson four: ML Basics with Keras"""

