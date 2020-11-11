import os
from zipfile import ZipFile

import requests
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Imports transfer learning weights and downloads it to a folder
transferurl = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
path = 'C:/Users/jes17/OneDrive/Documents/'
resptrain = requests.get(transferurl)
zname = os.path.join('/Users/jes17/OneDrive/Documents/datasets/Train/',
                     "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
zfile = open(zname, 'wb')
zfile.write(resptrain.content)
zfile.close()

# Imports transfer weights to the pre-trained model

local_weights_file = "C:/Users/jes17/OneDrive/Documents/datasets/Train/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Create the model
# InceptionV3 is used to call the pre trained model

# Arguments
# include_top - Boolean, whether to include the fully-connected layer at the top, as the last layer of the network. Default to True.
# weights - One of None (random initialization), imagenet (pre-training on ImageNet), or the path to the weights file to be loaded. Default to imagenet.
# input_tensor - Optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model. input_tensor is useful for sharing inputs between multiple different networks. Default to None.
# input_shape - Optional shape tuple, only to be specified if include_top is False (otherwise the input shape has
# pooling -	Optional pooling mode for feature extraction when include_top is False.None (default) means that the output of the model will be the 4D tensor output of the last convolutional block.avg means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor. max means that global max pooling will be applied.
# classes - optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified. Default to 1000.
# classifier_activiation - A str or callable. The activation function to use on the "top" layer. Ignored unless include_top=True. Set classifier_activation=None to return the logits of the "top" layer.

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

# Loads weights file to the model created above
pre_trained_model.load_weights(local_weights_file)

# Boolean, whether the layer's variables should be trainable.
# This method below takes the weights of the pre trained model, but train the new layers of the model without updating the weights of the pre trained model. This will allow the new output layers to learn to interpret the learned features of the new model.
for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()
# For example, perhaps you want to retrain some of the convolutional layers deep in the model, but none of the layers earlier in the model - this targets a specific layer for retrain
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Creates our dense model layers for output below the hidden layers on top from the transfer learning
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)

# Dropout Definition
# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
# Note that the Dropout layer only applies when training is set to True such that no values are dropped during inference. When using model.fit, training will be appropriately set to True automatically, and in other contexts, you can set the kwarg explicitly to True when calling the layer.
# (This is in contrast to setting trainable=False for a Dropout layer. trainable does not affect the layer's behavior, as Dropout does not have any variables/weights that can be frozen during training.)

# Arguments for Dropout
# rate - Float between 0 and 1. Fraction of the input units to drop.
# noise_shape - 1D integer tensor representing the shape of the binary dropout mask that will be multiplied with the input. For instance, if your inputs have shape (batch_size, timesteps, features) and you want the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).
# seed - A Python integer to use as random seed.

# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])

### Learning Rate Definition
# The amount that the weights are updated during training is referred to as the step size or the “learning rate.”
# Specifically, the learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value, often in the range between 0.0 and 1.0.
# During training, the backpropagation of error estimates the amount of error for which the weights of a node in the network are responsible. Instead of updating the weight with the full amount, it is scaled by the learning rate.
# This means that a learning rate of 0.1, a traditionally common default value, would mean that weights in the network are updated 0.1 * (estimated weight error) or 10% of the estimated weight error each time the weights are updated.

zip_file_url_train = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path = 'C:/Users/jes17/OneDrive/Documents/datasets/Cats and Dogs'

resptrain = requests.get(zip_file_url_train)

# Extracts the zip code file to the directory in datasets regular data
zname = os.path.join('/Users/jes17/OneDrive/Documents/datasets/Cats and Dogs', "cats_and_dogs_filtered.zip")
zfile = open(zname, 'wb')
zfile.write(resptrain.content)
zfile.close()
zf = ZipFile('/Users/jes17/OneDrive/Documents/datasets/Cats and Dogs/cats_and_dogs_filtered.zip')
# Extract its contents into <extraction_path>
# note that extractall will automatically create the path
zf.extractall(path='/Users/jes17/OneDrive/Documents/datasets/Cats and Dogs')
# close the ZipFile instance
zf.close()


class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.95):
            print("\nReached 80% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallBack()

base_dir = '/Users/jes17/OneDrive/Documents/datasets/Cats and Dogs/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # Directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # Directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # Directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # Directory with our validation dog pictures

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_steps=50,
    verbose=1,
    callbacks=[callbacks])

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
