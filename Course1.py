import tensorflow as tf
import pandas as pd
import numpy as np

# Imports the dataset
mnist = tf.keras.datasets.fashion_mnist
# Splits dataset into training and image files
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Goal of this task is to run a simple deep learning model from the datasets from mnist

# Data Discovery
#I want to find size and shape of data

print('Shape of Data: ', training_images.shape) # Prints the shape of the data
print('Highest Pixel Number of Data: ', training_images.max()) # Prints the highest value of an image
print('Lowest Pixel Number of Data: ', training_images.min()) # Prints the lowest value of an image
print('Length of Training Labels', len(np.unique(training_labels))) # Prints how many classifications there are for a label

# Shows what the Image looks like
import matplotlib.pyplot as plt
plt.imshow(training_images[5],cmap='gray')
print(training_labels[5])
print(training_images[5])

# Callbacks stops training when a certain accuracy is reached

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy')>0.95):
            print("\nReached 80% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallBack() #Passes in my callback function to callbacks to be called in fit method


#Normalization of image shape and color to take into account previouly it was just the image shape
#Creates a 4d list that includes the 28x28 size, the amount of images at 60000 and the color aspect of 1
#First convolution layer must map shape of data input as well as expect to import everything
#Convolution layers extract edges from images, so in the case of image classication it extracts important features from
#An image and helps better with model prediction
training_images = training_images.reshape((60000, 28, 28,1))
test_images = test_images.reshape((10000, 28, 28,1))


#Normalize the Images because we only want values of 0 and 1 for the modeel
training_images = training_images / 255.0
test_images = test_images / 255.0



# Build the Model

model = tf.keras.models.Sequential([
                                    # Sequential: That defines a SEQUENCE of layers in the neural network
                                    tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)),
                                    # (3,3) refers to the size in which the convoluation is
                                    # number of convolutions you want to generate. Purely arbitrary, but good to start with something in the order of 32
                                    # input shape refers to the shape of the data on input, must match the images for training and test
                                    # activation = relu activation function to use -- in this case we'll use relu, which you might recall is the equivalent of returning x when x>0, else returning 0
                                    tf.keras.layers.MaxPooling2D(3, 3),
                                    # MaxPooling layer which is then designed to compress the image, while maintaining the content of the features that were highlighted by the convlution.
                                    # the effect is to quarter the size of the image. Without going into too much detail here, the idea is that it creates a 2x2 array of pixels, and picks the biggest one, thus turning 4 pixels into 1
                                    tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(3,3),
                                    tf.keras.layers.Flatten(),
                                    # Flatten: Flatten just takes that square and turns it into a 1 dimensional set
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    # Dense: Adds a layer of neurons
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])


### Activation Syntax
# Relu effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.
# Softmax takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!

# Compiles the Model

model.compile(optimizer='AdaGrad', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
### Arguments
# optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, **kwargs


### Optimizers

# weight - Weights are used to connect the each neurons in one layer to the every neurons in the next layer. Weight determines the strength of the connection of the neurons

# RMSProp - Root Mean Square Propagation) is also a method in which the learning rate is adapted for each of the parameters. The idea is to divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight.[21] So, first the running average is calculated in terms of means square
# Adam - In this optimization algorithm, running averages of both the gradients and the second moments of the gradients are used
# SGD - algorithm in which the batch size is one. In other words, SGD relies on a single example chosen uniformly at random from a dataset to calculate an estimate of the gradient at each step
# AdaGrad - A sophisticated gradient descent algorithm that rescales the gradients of each parameter, effectively giving each parameter an independent learning rate

### Loss
# binarycrossentropy, binary_crossentropy  - Use this cross-entropy loss when there are only two label classes (assumed to be 0 and 1). For each example, there should be a single floating-point value per prediction.
# SparseCategoricalCrossentropy, sparse_categorical_crossentropy - Use this crossentropy loss function when there are two or more label classes. We expect labels to be provided as integers. If you want to provide labels using one-hot representation, please use CategoricalCrossentropy loss.
# CategoricalCrossentropy, categorical_crossentropy - Use this crossentropy loss function when there are two or more label classes. We expect labels to be provided in a one_hot representation.
# Mean Absolute Error, mean_absolute_error, tf.keras.losses.MeanAbsoluteError() -
# Mean Squared Error, mean_squared_error, tf.keras.losses.MeanSquaredError() -


# Fits the Model

model.fit(training_images, training_labels, epochs=5, callbacks = [callbacks], shuffle=True, verbose=1, steps_per_epoch=10)
### Arguments
#x, y
# batch_size(Integer or None. Number of samples per batch of computation. If unspecified, batch_size will default to 32.), sample_weight,
# callbacks(List of callbacks to apply during evaluation),
# verbose(0 or 1. Verbosity mode. 0 = silent, 1 = progress bar),
# workers(teger. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.)
# use_multiprocessing(If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.),
# max_que_size(	Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.),
# Epochs(Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.)
# validation_split()
# validation_data()
# shuffle(Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). This argument is ignored when x is a generator. 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.)
# class_weight(ptional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.)
# sample_weight(Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. This argument is not supported when x is a dataset, generator, or keras.utils.Sequence instance, instead provide the sample_weights as the third element of x.)
# initial_epoch(Integer. Epoch at which to start training (useful for resuming a previous training run).)
# steps_per_epoch(Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. If x is a tf.data dataset, and 'steps_per_epoch' is None, the epoch will run until the input dataset is exhausted. When passing an infinitely repeating dataset, you must specify the steps_per_epoch argument. This argument is not supported with array inputs.)
# validation_steps(Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch. If 'validation_steps' is None, validation will run until the validation_data dataset is exhausted. In the case of an infinitely repeated dataset, it will run into an infinite loop. If 'validation_steps' is specified and only part of the dataset will be consumed, the evaluation will start from the beginning of the dataset at each epoch. This ensures that the same validation samples are used every time.)
# validation_batch_size(Integer or None. Number of samples per validation batch. If unspecified, will default to batch_size. Do not specify the validation_batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches).)
# validation_freq(Only relevant if validation data is provided. Integer or collections_abc.Container instance (e.g. list, tuple, etc.). If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a Container, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.)



#Evalutes the Model

model.evaluate(test_images, test_labels)
### Arguments
# x, y,
# verbose(0 or 1. Verbosity mode. 0 = silent, 1 = progress bar),
# batch_size(Integer or None. Number of samples per batch of computation. If unspecified, batch_size will default to 32.), sample_weight,
# steps(Integer or None. Total number of steps (batches of samples) before declaring the evaluation round finished. Ignored with the default value of None. If x is a tf.data dataset and steps is None, 'evaluate' will run until the dataset is exhausted. This argument is not supported with array inputs),
# callbacks(List of callbacks to apply during evaluation),
# max_que_size(	Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.),
# return_dict(If True, loss and metric results are returned as a dict, with each key being the name of the metric. If False, they are returned as a list.),
# use_multiprocessing(If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.),
# workers(teger. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.)



classifications = model.predict(test_images)
print(classifications[1])
print(test_labels[1])


model.summary()


