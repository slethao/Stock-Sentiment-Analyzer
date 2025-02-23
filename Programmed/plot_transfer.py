import matplotlib.pyplot as plt
import tensorflow as tf
import io
import numpy as np
from PIL import Image

def plot_to_image(figure):
    figure.canvas.draw()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# def main():
#     plt.plot([1, 2, 3, 4], [5, 6, 7, 8])
#     fig = plt.gcf() 

#     writer = tf.summary.create_file_writer("logs") 
#     with writer.as_default():
#         tf.summary.image("matplotlib_plot", plot_to_image(fig), step=0)
#     """
#     tensorboard --logdir logs
#     """
# main()