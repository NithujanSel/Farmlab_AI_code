import pickle
from pickletools import optimize
from pyexpat import model
from statistics import mode
import time
X_train = pickle.load(open("X_train.pkl","rb"))
Y_train = pickle.load(open("Y_train.pkl","rb"))

X_test = pickle.load(open("X_test.pkl","rb"))
Y_test = pickle.load(open("Y_test.pkl","rb"))


from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,BatchNormalization,Dropout


from keras.callbacks import TensorBoard
NAME = f'Plant model perdict {int(time.time())}'
tb = TensorBoard(log_dir=f'logs/{NAME}')


# import tensorflow as tf
# from tensorflow.python.framework.config import set_memory_growth
# tf.compat.v1.disable_v2_behavior()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)


model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape = X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#output
model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(3,activation='softmax'))

#model.summary()


model.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=['accuracy'])

#model.fit(X,Y,epochs=1,validation_split = 0.3,batch_size=12,callbacks= [tb])
model.fit(X_train,Y_train,epochs=50,validation_data = (X_test,Y_test),callbacks= [tb])

model.save("plantmodel.h5")