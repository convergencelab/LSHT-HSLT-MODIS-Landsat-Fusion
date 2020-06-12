"""
implementation of the Layer used to build Residual blocks
"""

import tensorflow as tf


class ResLayer(tf.keras.layers.Layer):
  def __init__(self, kernel_size, filters, momentum, strides):
    super(ResLayer, self).__init__()
    filters1, filters2 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides)
    self.bn2a = tf.keras.layers.BatchNormalization(momentum=momentum)

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', strides=strides)
    self.bn2b = tf.keras.layers.BatchNormalization(momentum=momentum)

  def call(self, input_tensor, training=False):
      x = self.conv2a(input_tensor)
      x = self.bn2a(x, training=training)
      x = tf.nn.relu(x)

      # skip activation on last one #
      x = self.conv2b(x)
      x = self.bn2b(x, training=training)

      x += input_tensor
      return tf.nn.relu(x)