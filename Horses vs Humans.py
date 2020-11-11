import os
from zipfile import ZipFile

import matplotlib.pyplot as plt
import requests
import tensorflow as tf

# Imports the data from an url
zip_file_url_train = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
zip_file_url_val = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'
path = 'C:/Users/jes17/OneDrive/Documents/datasets/Train/'

resptrain = requests.get(zip_file_url_train)
respval = requests.get(zip_file_url_val)

# Extracts the zip code file to the directory in datasets regular data
zname = os.path.join('/Users/jes17/OneDrive/Documents/datasets/Train/', "horse-or-human.zip")
zfile = open(zname, 'wb')
zfile.write(resptrain.content)
zfile.close()
zf = ZipFile('/Users/jes17/OneDrive/Documents/datasets/Train/horse-or-human.zip')
# Extract its contents into <extraction_path>
# note that extractall will automatically create the path
zf.extractall(path='/Users/jes17/OneDrive/Documents/datasets/Train')
# close the ZipFile instance
zf.close()

# Extracts the zip code file to the directory in datasets validation data
zname = os.path.join('/Users/jes17/OneDrive/Documents/datasets/Validation/', "validation-horse-or-human.zip")
zfile = open(zname, 'wb')
zfile.write(respval.content)
zfile.close()
zf = ZipFile('/Users/jes17/OneDrive/Documents/datasets/Validation/validation-horse-or-human.zip')
# Extract its contents into <extraction_path>
# note that extractall will automatically create the path
zf.extractall(path='/Users/jes17/OneDrive/Documents/datasets/Validation')
# close the ZipFile instance
zf.close()

# Directory with our training horse pictures
train_horse_dir = os.path.join('/Users/jes17/OneDrive/Documents/datasets/Train/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('/Users/jes17/OneDrive/Documents/datasets/Train/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('/Users/jes17/OneDrive/Documents/datasets/Validation/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('/Users/jes17/OneDrive/Documents/datasets/Validation/humans')

# Shows length to make sure they match
print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
print('total validation horse images:', len(os.listdir(validation_horse_dir)))
print('total validation human images:', len(os.listdir(validation_human_dir)))

# Names of Files

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])


# Callback Method to Stop training once significant accuracy is reached
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.95):
            print("\nReached 80% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallBack()  # Passes in my callback function to callbacks to be called in fit method

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Shows summary of the model
model.summary()

# Compiles the model
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

### With Images in a file directory, we have to set up a generator to read images and rescale them and feed them through the neural network in batches.
# This is seperate from a loading data through a parameter that contains the images
# Class mode must match the loss implemented in the model.sequtial statement

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1 / 255)

### Arguments
# featurewise_center (Boolean. Set input mean to 0 over the dataset, feature-wise.)
# samplewise_center (Boolean. Set each sample mean to 0.)
# featurewise_std_normalization (Boolean. Divide inputs by std of the dataset, feature-wise.)
# samplewise_std_normalization (Boolean. Divide each input by its std.)
# zca_epsilon (epsilon for ZCA whitening. Default is 1e-6.)
# zca_whitening (Boolean. Apply ZCA whitening.)
# rotation_range (Int. Degree range for random rotations.)
# width_shift_range ()
# height_shift_range ()
# brightness_range (Tuple or list of two floats. Range for picking a brightness shift value from.)
# shear_range (Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees))
# zoom_range (Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].)
# channel_shift_range (Float. Range for random channel shifts.)
# fill_mode (One of {"constant", "nearest", "reflect" or "wrap"}. Default is 'nearest'. Points outside the boundaries of the input are filled according to the given mode:
# 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
# 'nearest': aaaaaaaa|abcd|dddddddd
# 'reflect': abcddcba|abcd|dcbaabcd
# 'wrap': abcdabcd|abcd|abcdabcd)
# cval (Float or Int. Value used for points outside the boundaries when fill_mode = "constant".)
# horizontal_flip (Boolean. Randomly flip inputs horizontally.)
# vertical_flip (Boolean. Randomly flip inputs vertically.)
# rescale (rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (after applying all other transformations).)
# preprocessing_function (	function that will be applied on each input. The function will run after the image is resized and augmented. The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape.)
# data_format (Image data format, either "channels_first" or "channels_last". "channels_last" mode means that the images should have shape (samples, height, width, channels), "channels_first" mode means that the images should have shape (samples, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".)
# validation_split (Float. Fraction of images reserved for validation (strictly between 0 and 1).)
# dtype (Dtype to use for the generated arrays.)


# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    '/Users/jes17/OneDrive/Documents/datasets/Train/',
    # Directory -  This is the source directory for training images
    target_size=(150, 150),
    # All images will be resized to 150x150 Tuple of integers (height, width), defaults to (256,256). The dimensions to which all images found will be resized.
    batch_size=128,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary'
)

### Arguments flow from directory
# color_mode (One of "grayscale", "rgb", "rgba". Default: "rgb". Whether the images will be converted to have 1, 3, or 4 channels.)
# classes (Optional list of class subdirectories (e.g. ['dogs', 'cats']). Default: None. If not provided, the list of classes will be automatically inferred from the subdirectory names/structure under directory, where each subdirectory will be treated as a different class (and the order of the classes, which will map to the label indices, will be alphanumeric). The dictionary containing the mapping from class names to class indices can be obtained via the attribute class_indices.)
# class_ mode(One of "categorical", "binary", "sparse", "input", or None. Default: "categorical". Determines the type of label arrays that are returned: - "categorical" will be 2D one-hot encoded labels, - "binary" will be 1D binary labels, "sparse" will be 1D integer labels, - "input" will be images identical to input images (mainly used to work with autoencoders). - If None, no labels are returned (the generator will only yield batches of image data, which is useful to use with model.predict_generator()). Please note that in case of class_mode None, the data still needs to reside in a subdirectory of directory for it to work correctly.)
# shuffle (Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.)
# seed (Optional random seed for shuffling and transformations.)
# save_to_dir (	None or str (default: None). This allows you to optionally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).)
# save_prefix (Str. Prefix to use for filenames of saved pictures (only relevant if save_to_dir is set).)
# save_format (	One of "png", "jpeg" (only relevant if save_to_dir is set). Default: "png".)
# follow_links (Whether to follow symlinks inside class subdirectories (default: False).)
# subset (Subset of data ("training" or "validation") if validation_split is set in ImageDataGenerator.)
# interpolation (Interpolation method used to resample the image if the target size is different from that of the loaded image. Supported methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3 or newer is installed, "lanczos" is also supported. If PIL version 3.4.0 or newer is installed, "box" and "hamming" are also supported. By default, "nearest" is used.)


# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    '/Users/jes17/OneDrive/Documents/datasets/Validation/',  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=32,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# fit.generator has same parameters as model.fit
history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8,
    callbacks=[callbacks]
)

# Retrieve a list of list results on training and test data
# sets for each training epoch

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.legend(loc=0)
plt.figure()

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.legend(loc=0)
plt.show()
