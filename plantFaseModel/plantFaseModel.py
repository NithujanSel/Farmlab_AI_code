#Nodige libirary in laden
import pandas as pd 
import os 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator

#De trian data set in laden en dat verdelen tussen train en validatie set
train_path = "./train/"

#train set
train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_path,
                                                                 label_mode="categorical", 
                                                                 shuffle=True,
                                                                 image_size=(224,224),
                                                                batch_size = 32,
                                                                 seed=42,
                                                                validation_split=0.2,
                                                                   subset="training")

#validatie set
valid_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_path,
                                                                 label_mode="categorical", 
                                                                 shuffle=False,
                                                                 image_size=(224,224),
                                                                batch_size = 32,
                                                                 seed=42,
                                                                validation_split=0.2,
                                                                   subset="validation")

#Model maken voor onze dataset

class_names = train_data.class_names #De fase naam opslaan.

base_model = tf.keras.applications.EfficientNetB0(include_top= False) #EfficientNetB0 cnn model in laden

base_model.trainable = False 

inputs = tf.keras.layers.Input(shape=(224,224,3), name='Input_Layer') #Laag dimenstie meegeven

x = base_model(inputs) 

x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x) #GemiddelPooling van conv laag

outputs = tf.keras.layers.Dense(len(class_names), activation='softmax', name='output_layer')(x)#De outpult laag

model = tf.keras.Model(inputs, outputs)

model.compile(loss='categorical_crossentropy',
             optimizer = tf.keras.optimizers.Adam(), 
             metrics=['accuracy']) 

#Model runnen
model_history = model.fit(train_data, 
                          steps_per_epoch=len(train_data),
                          validation_data= valid_data,
                        validation_steps= len(valid_data),
                         epochs=5
                         )


#Model opslaan
model.save("plantModelFaseV4.h5")

model.evaluate(valid_data)



































#https://www.kaggle.com/code/atrisaxena/using-tensorflow-2-x-classify-plant-seedlings