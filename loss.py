import tensorflow as tf
from gram_matrix import gram_matrix


# Define the content loss function
class ContentLoss(tf.keras.losses.Loss):
    def __init__(self, target):
        super().__init__()
        self.target = target
    def call(self, input):
        loss = tf.reduce_mean((input - self.target) ** 2)
        return loss

# Define the style loss function
class StyleLoss(tf.keras.losses.Loss):
    def __init__(self, target):
        super().__init__()
        self.target = target
    def call(self, input):
        G = gram_matrix(input)
        loss = tf.reduce_mean((G - self.target) ** 2)
        return loss