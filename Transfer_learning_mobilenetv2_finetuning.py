import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset_path_new = ""

#train_dir = '\training_set'
#validation_dir = "\test_set"
IMG_SHAPE = (128, 128, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

base_model.trainable = False


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

prediction_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)

model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)


train_generator = data_gen_train.flow_from_directory('dataset/training_set', target_size=(128,128), batch_size=32, class_mode="binary")

valid_generator = data_gen_valid.flow_from_directory('dataset/test_set', target_size=(128,128), batch_size=32, class_mode="binary")

model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

#Acc-92.56 val-93.20

base_model.trainable = True

fine_tune_at = 100
for layer in base_model.layers[0:fine_tune_at]:
    layer.trainable = False
    
#Compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)


