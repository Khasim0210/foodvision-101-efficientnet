import tensorflow as tf

IMAGE_SIZE = (224, 224)

import tensorflow as tf
IMAGE_SIZE = (224, 224)

def preprocess_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label
