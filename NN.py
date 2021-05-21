# %matplotlib inline
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryCrossentropy
from keras.regularizers import l2
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(validation_split=0.1,
                            rescale = 1./255,
                            horizontal_flip=True,
                            zoom_range=0.2,
                            shear_range=0.2)

testgen = ImageDataGenerator(rescale = 1./255)


train_it = datagen.flow_from_directory(
    'Data/training_set/',
    target_size = (16,16),
    class_mode='binary',
    batch_size=64,
    subset = 'training',
    color_mode = 'grayscale',
    shuffle=True)

val_it = datagen.flow_from_directory(
    'Data/training_set/',
    target_size = (16,16),
    class_mode='binary',
    batch_size=64,
    subset = 'validation',
    color_mode = 'grayscale',
    shuffle=True)

test_it = testgen.flow_from_directory(
    'Data/test_set/',
    target_size = (16,16),
    class_mode='binary',
    color_mode = 'grayscale',
    batch_size=64,
    shuffle=False)



model_NN = keras.Sequential(
    [
        layers.Flatten(),
        layers.Dense(512,activation='relu', name = 'layer1'),
        layers.Dense(512,activation='relu', name = 'layer2'),
        layers.Dense(256,activation='relu', name = 'layer3'),
        layers.Dense(1,activation='sigmoid', name = 'layer7')
    ]
)


model_NN.build((None,16,16,1))
model_NN.summary()


model_NN.compile(optimizer=Adam(learning_rate=0.001),
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

history_NN = model_NN.fit(x=train_it,
                        validation_data=val_it,
                        epochs=60,
                        verbose=1,
                        shuffle=True)

model_NN.evaluate(test_it)

model_NN.save("model")

plt.plot(history_NN.history['accuracy'])
plt.plot(history_NN.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_NN.history['loss'])
plt.plot(history_NN.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
