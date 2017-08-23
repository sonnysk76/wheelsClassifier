import numpy as np
import glob
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam, Adadelta
from keras.utils import np_utils
from keras.losses import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator
import theano


os.chdir("/home/salvador/Documents/170520w/comprimidos/")
#lista = glob.glob('*.bmp')
lista = pd.read_csv('/home/salvador/Documents/170520w/lista.csv')
lista = lista['VIN']


imArray = np.array([np.array(Image.open("/home/salvador/Documents/170520w/comprimidos/"+str(foto)+".bmp")).flatten() for foto in lista], 'f')
#imArray = np.array(Image.open("1.bmp"))




listado = pd.read_csv('/home/salvador/Documents/170520w/lista.csv')
label = listado['LABEL']
etiquetas = np.array(label)

data, Labels = shuffle(imArray, etiquetas, random_state = 2)

train_data = [data, Labels]
print (train_data[0].shape)
print (train_data[1].shape)
imag = imArray[167].reshape(200, 200, 3)
plt.imshow(imag, cmap='gray')
plt.title(etiquetas[167])
plt.show()

batch_size = 32
nb_classes = 26
nb_epochs = 50
img_rows = 200
img_cols = 200
img_channels = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

(X, y) = (train_data[0], train_data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("X Train shape ",X_train.shape)
print(X_train.shape[0], "Muestras Entrenar")
print(X_test.shape[0], "Muestrar prueba")
#print("Ytest ", y_test)

y_train = np_utils.to_categorical(y_train, num_classes=None)
y_test = np_utils.to_categorical(y_test, num_classes=26)
#print("Ytest ", y_test)
#plt.imshow(X_train[100], interpolation='nearest')
#plt.show()

# modelo de red
#print(imag.shape)

def generator(X_train,y_train, n ):
    while True:
        Xn = []
        yn = []
        for i in range(n):
            idx = randint(0, len(X_train)-1)
            Xi, yi = image_array(X_train[idx], y_train[idx])
            Xn.append(Xi)
            yn.append(yi)
        yield (np.asarray(Xn), np.asarray(yn))

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(200, 200, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(26))
model.add(Activation('softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer= Adadelta(),
              metrics=['accuracy'])



model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

hist = model.fit(X_train, y_train, batch_size=16, epochs=50, verbose=1, validation_data=(X_test, y_test))

model_json = model.to_json()
with open("/home/salvador/PycharmProjects/IMClassifier/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("/home/salvador/PycharmProjects/IMClassifier/model.h5")

model.save('/home/salvador/PycharmProjects/IMClassifier/ClasAugm.h5')
#14/Agosto/17
#4138/4138 [==============================] - 17s - loss: 0.1776 - acc: 0.9430 - val_loss: 0.0630 - val_acc: 0.9865


