# Define a function to extract the features from the input image
# using the 
def get_features(image, model, layers):
    features = {}
    x = image
    for layer in model.layers:
        x = layer(x)
        if layer.name in layers:
            features[layer.name] = x
    return features
