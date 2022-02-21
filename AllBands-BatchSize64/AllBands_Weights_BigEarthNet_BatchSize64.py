# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub                    #saved models in tf hub
import pdb

def start_session(GPU, MEMORY_LIMIT=0.4):

    import os
    import tensorflow as tf
    from tensorflow.python.keras.backend import set_session

    # GPU = "1" # 0->GTX1080Ti, 1->GTX2080Ti, None
    if GPU:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=GPU
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    
    config = tf.compat.v1.ConfigProto()
    if MEMORY_LIMIT>0 and MEMORY_LIMIT<1:
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.gpu_options.per_process_gpu_memory_fraction=MEMORY_LIMIT # limiting the maximum memory
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # Set this TensorFlow session as the default session for Keras
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    set_session(sess) 


GPU = "0" # 0->GTX1080Ti, 1->GTX2080Ti
MEMORY_LIMIT = 1

start_session(GPU, MEMORY_LIMIT)

DATA_DIR = "./eurosat/all"


(train, val, test), ds_info = tfds.load("eurosat/all", 
                               with_info=True,
                               split=["train[:60%]", "train[60%:70%]", "train[70%:]"], #Train, val, test = 60, 10, 30
                               data_dir=DATA_DIR)

print(ds_info)

"""## Exploring the dataset


"""

ds_info.features

class_names = []
for i in range(ds_info.features["label"].num_classes):
  class_names.append(ds_info.features["label"].int2str(i))

class_names

#This dataset contains only one split train with 27000 images with 10 classes
list(ds_info.splits.keys())

#The data is in a dictionary with three keys: 'filename', 'image(as array)' and the 'label' 
datapoint = next(iter(train)) #iterating over the dataset
datapoint

"""## Data Augmentation"""

#Folder to save the best model
FILE_MODEL = "./best_model/Resnet50BigEarthNetAll.hdf5"

#Define main variables
NUM_EPOCHS = 100
#BATCH_SIZE = 128
BATCH_SIZE = 64
BUFFER_SIZE = 1000

IMAGE_SHAPE = [224, 224, 3] #RGB for the pre-trained CNN
NUM_CLASSES = ds_info.features["label"].num_classes

#As the dataset is batched is better to initialize:
STEPS_PER_EPOCH = int(ds_info.splits["train"].num_examples * 0.6)//BATCH_SIZE
VALIDATION_STEPS = int(ds_info.splits["train"].num_examples * 0.1)//BATCH_SIZE

def data_preprocessing(datapoint):
  input_image = tf.image.resize(datapoint["sentinel2"], size=([IMAGE_SHAPE[0], IMAGE_SHAPE[1]])) #Size must be 1D-tensor of two elements: width and height
  
  input_image = input_image/65535           #Normalize uint16 values between 0 and 1
      
  return input_image, datapoint["label"]

#Prepare data
train_dataset = train.map(data_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE) #num_parallel_call is used for number of processors
validation_dataset = val.map(data_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test.map(data_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#Batch the data and use caching & prefetching to optimize loading speed
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#The data is in a dictionary with three keys: 'filename', 'image(as array)' and the 'label' 
datapoint = next(iter(train_dataset)) #iterating over the dataset
datapoint

"""## Parameters configuration and data preprocessing

### Import, build and compile the model (ResNet50)
"""

#ResNet50
#Import ResNet50 pre-trained on BigEarthNet
hub_url = "https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1"  
hub_layer = hub.KerasLayer(hub_url, input_shape=IMAGE_SHAPE, trainable=False)

#Keras API to build last layer and the final model
model_resnet = tf.keras.Sequential([
                                    tf.keras.layers.Input(shape=(224, 224, 13)),
                                    # introduce a additional layer to get from 13 to 3 input channels
                                    tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1)),
                                    hub_layer, 
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model_resnet.build((None, 224, 224, 3))
model_resnet.summary()

#Compile, define loss, optimizers and metrics
#Labels are not one-hot encoded so we need to use sparse categorical cross-entropy
model_resnet.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy']
)

#Stop training and save the best model

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

callback = [
    tf.keras.callbacks.ModelCheckpoint(filepath=FILE_MODEL, monitor='val_accuracy', verbose=1, mode='max', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, min_delta=1e-6,verbose=True, mode='min')
]

"""### Train the model"""

#Train the models [ResNet]
history_res = model_resnet.fit(
    train_dataset,
    epochs = NUM_EPOCHS,
    steps_per_epoch = STEPS_PER_EPOCH,
    callbacks = callback,
    validation_data = validation_dataset,
    validation_steps = VALIDATION_STEPS
)

"""
References

Codes adapted from:
1. CodeX For ML, 18 Sept. 2020, "How to use TensorFlow Datasets? Image classification with EuroSAT dataset with TFDS", [Video], YouTube, URL: https://www.youtube.com/watch?v=6th3rahsw9Y.
2. Vera, Adonaí. 1 Dec. 2021, “Curso Profesional De Redes Neuronales Con Tensorflow.” [E-Learning Website], URL: https://platzi.com/cursos/redes-neuronales-tensorflow/. 
3. Jens Leitloff and Felix M. Riese, "Examples for CNN training and classification on Sentinel-2 data", Zenodo, 10.5281/zenodo.3268451, 2018. [Repository], URL: https://github.com/jensleitloff/CNN-Sentinel


Tensorflow models and datasets:
1. Maxim Neumann, Andre Susano Pinto, Xiaohua Zhai,and Neil Houlsby,   “In-domain representation learningfor remote sensing,” Nov. 2019. URL: https://tfhub.dev/google/collections/remote_sensing/1

"""