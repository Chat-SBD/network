import tensorflow as tf

def gradient(model, lossf, data):
    """
    Find a gradient on a model with some data.

    Args:
        model: tensorflow.keras.Model. The model to find the gradients on.
        lossf: tensorflow.keras.Loss. The loss function to use.
        data: tuple. (video frames, number of white lights).
    """
    frames, lights = model
    with tf.GradientTape() as tape:
        preds = model(frames, training = True)
        loss = lossf(lights, preds)
    
    return tape.gradient(loss, model.trainable_weights)