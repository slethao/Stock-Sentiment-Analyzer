import matplotlib.pylab
import matplotlib.pyplot
import tensorflow
import os
import re
import shutil
import string
import pathlib
from tensorflow.keras import layers
from tensorflow.keras import losses
import matplotlib

url_link = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tensorflow.keras.utils.get_file("aclImdb_v1", url_link, untar = True, cache_dir = "./")
dataset_pathway = pathlib.Path("datasets/aclImdb_v1/aclImdb")
print(os.listdir(dataset_pathway)) # the content groups (main subdirectories) in the directory is ['imdb.vocab', 'imdbEr.txt', 'README', 'test', 'train']

training_path = os.path.join(dataset_pathway, "train") # all the content in the file called 'train'
print(os.listdir(training_path))

going_to_be_read = os.path.join(training_path, "pos/1181_9.txt")
with open(going_to_be_read) as file:
    print(file.read())

"""Load the dataset"""
remove_directory = os.path.join(training_path, "unsup")
shutil.rmtree(remove_directory)

batch_size = 32
seed = 42

raw_train_dataset = tensorflow.keras.utils.text_dataset_from_directory(
                                                                        f"{dataset_pathway}/train",
                                                                        batch_size = batch_size,
                                                                        validation_split = 0.2,
                                                                        subset = 'training',
                                                                        seed = seed) #NOTE the 'text_dataset_from_directory' is a function that is used to load the text data from the directory                                                               

raw_validation_dataset = tensorflow.keras.utils.text_dataset_from_directory(
                                                                            f"{dataset_pathway}/train",
                                                                            batch_size = batch_size,
                                                                            validation_split = 0.2,
                                                                            subset = 'validation',
                                                                            seed = seed)

raw_test_dataset = tensorflow.keras.utils.text_dataset_from_directory(f"{dataset_pathway}/test", 
                                                                      batch_size = batch_size)
"""
Standardization is used to preproccess the text
(in this case you are going to remove the HTML 
elemetns to simplify the dataset) 

Tokenization refers to splitting strings into 
tokens (spliting (using the spaces) a sentence 
into indicicual words)

Vextorization refer to converting tokening tokens 
into numbers so they can be fed into a neural 
netowrk.

"""
def custom_standardization(input_data):
    lowercase = tensorflow.strings.lower(input_data) # 'DType' object has no attribute 'lower'
    stripped_html = tensorflow.strings.regex_replace(lowercase, '<br />', ' ')
    return tensorflow.strings.regex_replace(stripped_html,
                                            '[%s]' % re.escape(string.punctuation), 
                                            '') 

max_features = 10000 
sequence_length = 250

vector_layer = layers.TextVectorization(
    standardize = custom_standardization,
    max_tokens = max_features,
    output_mode = 'int',
    output_sequence_length = sequence_length
)

"""
each toen that was converted into an index references a word
1287 --->  silent
313 --->  night

    using the .get_vocabulary() method to see the top 5 words
"""
train_txt = raw_train_dataset.map(lambda x, y: x)
vector_layer.adapt(train_txt)

def Vextorization_txt(txt, label):
    txt = tensorflow.expand_dims(txt, -1)
    return vector_layer(txt), label

txt_batch, label_batch = next(iter(raw_train_dataset))
first_review, first_label = txt_batch[0], label_batch[0]
print(f"Review {first_review}")
print(f"Label {raw_train_dataset.class_names[first_label]}")
print(f"Vectorized review {vector_layer(first_review)}")

train_dataset = raw_train_dataset.map(Vextorization_txt)
val_dataset = raw_validation_dataset.map(Vextorization_txt)
test_dataset = raw_test_dataset.map(Vextorization_txt)

"""Configure the dataset for performance"""
# .cache() allows you to keep dataset

# AUTOTUNE = tensorflow.data.AUTOTUNE is within the .prefetch()
# .prefetch() lets you do data preproccessing and model execution with training

AUTOTUNE = tensorflow.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size = AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size = AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size = AUTOTUNE)

"""Build and Train Model"""
embedding_dim = 16 # vector space used to contain a word
model = tensorflow.keras.Sequential([layers.Embedding(max_features + 1, embedding_dim),
                                     layers.Dropout(0.2),
                                     layers.GlobalAveragePooling1D(),
                                     layers.Dropout(0.2),
                                     layers.Dense(1, activation = 'sigmoid')])

model.summary() # actually prints out a chart

model.compile(loss = losses.BinaryCrossentropy(),
              optimizer = 'adam',
              metrics = [tensorflow.metrics.BinaryAccuracy(threshold = 0.5)])

# training is done here
epochs = 10
history = model.fit(train_dataset, 
                    validation_data = val_dataset, 
                    epochs = epochs)

"""
Conclusion
"""
losses_made, confidence = model.evaluate(test_dataset)
print(f"Loss: {losses_made}")
print(f"Accuracy: {confidence}")

"""NOTE: Create a plot of accuracy and loss over time (important)"""
history_set = history.history
print(history_set.keys()) # all the types of losses and accuracies

accuracy = history_set['binary_accuracy']
value_accuracy = history_set['val_binary_accuracy']
loss = history_set['loss']
value_loss = history = history_set['val_loss']

epochs = range(1, len(accuracy) + 1)

matplotlib.pyplot.plot(epochs, loss, 'bo', label = 'Training loss')
matplotlib.pyplot.plot(epochs, value_loss, 'b', label = 'Validation loss')
matplotlib.pyplot.title('Traingin and validation loss')
matplotlib.pyplot.xlabel('Epochs')
matplotlib.pyplot.ylabel('Loss')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

matplotlib.pyplot.plot(epochs, accuracy, 'bo', label = 'Training accuracy')
matplotlib.pyplot.plot(epochs, value_accuracy, 'b', label = 'Validation accuracy')
matplotlib.pyplot.title('Training and validation accuracy')
matplotlib.pyplot.xlabel('Epochs')
matplotlib.pyplot.ylabel('Accuracy')
matplotlib.pyplot.legend(loc='lower right')
matplotlib.pyplot.show()

# export the model
exported_model = tensorflow.keras.Sequential([vector_layer, 
                                              model, 
                                              layers.Activation('sigmoid')])

exported_model.compile(loss = losses.BinaryCrossentropy(from_logits = False), 
                       optimizer = "adam", 
                       metrics = ['accuracy'])

#testing to see if it works
metrics = exported_model.evaluate(raw_test_dataset, return_dict = True)
print(metrics)

#NOTE interfence on new data
example = tensorflow.constant(["The movie was great!",
                               "The movie was okay.",
                               "The movie was terribel..."])
exported_model.predict(example)
"""
exported_model.predict(example)

Output:
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 148ms/step
array([[0.57763124],
       [0.5442707 ],
       [0.53183067]], dtype=float32)

       The movie was great! --> 58% was voted 
       The movie was okay --> 54% was voted
       The movie was terrible... --> 53% was voted

"""