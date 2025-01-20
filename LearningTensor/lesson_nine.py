import tensorflow
import datetime

"""
using tensorboard

tensorboard --logdir logs/fit
"""

data_set = tensorflow.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data_set.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    return tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
        tensorflow.keras.layers.Dense(128, activation='relu'),
        tensorflow.keras.layers.Dropout(0.2),
        tensorflow.keras.layers.Dense(10)
    ])

model = create_model()
model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, y=y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
