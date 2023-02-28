import tensorflow as tf
import numpy as np
import PIL.Image
from training_loop import train_step
from get_features import get_features
from loss import ContentLoss, StyleLoss
from gram_matrix import gram_matrix

# Load the pre-trained VGG-19 network
# 19 Layers 16 convolutional 3 fully conected
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Freeze all layers
# values are fixed during training and not updated by the optimizer
for layer in vgg.layers:
    layer.trainable = False

# Define the layers to extract content and style features
# These refer to layers from the VGG-19 NN
# One content layer
# Five style layers
content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Load the content and style images
content_image = np.array(PIL.Image.open("content.png"))
style_image = np.array(PIL.Image.open("style.jpg"))

# Define the image transformations
# Preprocesses an input image in the format expected by the VGG-19 model
preprocess = tf.keras.applications.vgg19.preprocess_inputrain_stept

# Apply the transformations to the images
content_tensor = preprocess(content_image).astype('float32')
style_tensor = preprocess(style_image).astype('float32')

# Define the input image as a copy of the content image
input_tensor = tf.Variable(content_tensor)

# Define the content and style features
content_features = get_features(content_tensor, vgg, content_layers)
style_features = get_features(style_tensor, vgg, style_layers)

# Compute the target style features
# relative importance assigned to the individual layers of the CNN
style_weights = [1.0, 0.75, 0.5, 0.25, 0.1]
target_features = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# Define the content and style loss functions
content_weight = 1
style_weight = 1000
content_loss = ContentLoss(content_features['block4_conv2'])
style_loss = tf.reduce_mean([StyleLoss(target_features[layer])(style_features[layer]) * weight for layer, weight in zip(style_layers, style_weights)])

# Define the total loss function
total_loss = content_weight * content_loss + style_weight * style_loss

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Define the training loop
def train_step(image, style_reference, content_weight, style_weight, optimizer):
    with tf.GradientTape() as tape:
        # Calculate the loss
        total_loss = content_weight * content_loss + style_weight * style_loss

    # Calculate the gradients of the loss with respect to the image
    gradients = tape.gradient(total_loss, image)

    # Apply the gradients to the image
    optimizer.apply_gradients([(gradients, image)])

    return total_loss