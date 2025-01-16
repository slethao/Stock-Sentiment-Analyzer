"""
Text Classification with TensorFlow
Example: Movie Review
"""
import tensorflow
import tensorflow_hub
import tensorflow_datasets

# Download the IMDB dataset
training, validation = tensorflow_datasets.load(
                                                        name = "imdb_reviews",
                                                        split=('train[:60%]', 'train[60%:80%]', 'train[80%:]'),
                                                        as_supervised = True,
                                                        with_info=True)
train, valid, test = training

train_exp_batch, train_labels_batch = next(iter(train.batch(10)))
print(train_exp_batch)
print(train_labels_batch)

# Build model and train it!!
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = tensorflow_hub.KerasLayer(embedding, 
                                      input_shape = [],
                                      dtype = tensorflow.string,
                                      trainable=True)
hub_layer(train_exp_batch[:3])
#build time
build_model = tensorflow.keras.Sequential()
build_model.add(hub_layer)
build_model.add(tensorflow.keras.layers.Dense(16, activation = 'relu'))
build_model.add(tensorflow.keras.layers.Dense(1))
build_model.summary()
build_model.compile(optimizer = 'adam',
                    loss = tensorflow.keras.losses.BinaryCrossentropy(from_logits = True),
                    metrics = ['accuracy'])
#training the model
train_model = build_model.fit(training.shuffle(10000).batch(512),
                              epochs = 10,
                              validation_data = validation.batch(512),
                              verbose = 1)

# feb 25th

#Conclusion
results = build_model.evaluate(test.batch(512), verbose = 2)
for name, value in zip(build_model.metrics_name, results):
    print("%s: %3f" % (name, value))