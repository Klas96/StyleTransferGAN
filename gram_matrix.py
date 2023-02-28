import tensorflow as tf

# Define a function to compute the gram matrix of a tensor
def gram_matrix(tensor):
    shape = tf.shape(tensor)
    features = tf.reshape(tensor, [shape[0], shape[1] * shape[2], shape[3]])
    gram = tf.matmul(features, features, transpose_b=True)
    return gram / tf.cast(shape[1] * shape[2] * shape[3], tf.float32)
