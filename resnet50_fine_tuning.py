from google.colab import drive
drive.mount('/content/drive')

!unzip -qq '__PATH_TO_GDRIVE_FOLDER__/digits_small.zip'
!unzip -qq '__PATH_TO_GDRIVE_FOLDER__/cwe3categ.zip'
!unzip -qq '__PATH_TO_GDRIVE_FOLDER__/cwe3categ_augmented.zip'
!unzip -qq '__PATH_TO_GDRIVE_FOLDER__/cwe_dataset2.zip'

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback
from keras import backend, Input, Model
from keras.applications import ResNet50

from keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

class AccuracyCallback(Callback):
    def __init__(self, test_data):
        self.imageflow = test_data
        
    def on_epoch_end(self, epoch, logs=None):
      Y_pred = model.predict(self.imageflow)
      y_pred = np.argmax(Y_pred, axis=1)
      print(confusion_matrix(self.imageflow.classes, y_pred))
      

backend.clear_session()

batch_size = 20
epochs = 20
image_shape = (330, 330, 3)
#image_shape = (32, 32, 3)

#train_path = '/content/_splitted/train'
#train_path = '/content/digits_small/train'
#train_path = '/content/cwe3categ_augmented/train'
train_path = '/content/cwe_dataset2/train'
#train_path = '__PATH_TO_GDRIVE_FOLDER__/cwe_dataset/train'

#validation_path = '/content/_splitted/validation'
#validation_path = '/content/digits_small/validation'
#validation_path = '/content/cwe3categ_augmented/validation'
validation_path = '/content/cwe_dataset2/validation'

conv_base = ResNet50(
    weights='imagenet', 
    include_top=False, 
    input_tensor=Input(shape=image_shape))


# 1. nacteni dat a jejich prepocet dle konvoluce z VGG16
train_imagedatagen = ImageDataGenerator(#rescale=1./255, 
                                  rotation_range=30,
                                  zoom_range=0.1,
                                  horizontal_flip=True,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  fill_mode='nearest')

validation_imagedatagen = ImageDataGenerator()
# prumerne hodnoty (z ImageNet) pro preprocessing
train_imagedatagen.mean = np.array([123.68, 116.779, 103.939], dtype='float32')
validation_imagedatagen.mean = np.array([123.68, 116.779, 103.939], dtype='float32')

train_imageflow = train_imagedatagen.flow_from_directory(
      train_path, 
      color_mode= 'rgb', 
      batch_size=batch_size,
      class_mode='categorical',
      target_size = image_shape[:2])
  
train_counter = Counter(train_imageflow.classes)
print(train_counter.items())
train_class_dict = train_imageflow.class_indices
train_sample_count = train_imageflow.samples

validation_imageflow = validation_imagedatagen.flow_from_directory(
      validation_path,
      color_mode= 'rgb', 
      batch_size=batch_size,
      class_mode='categorical',
      target_size = image_shape[:2])
  
validation_counter = Counter(validation_imageflow.classes)
print(validation_counter.items())
validation_class_dict = validation_imageflow.class_indices
validation_sample_count = validation_imageflow.samples

head_model = conv_base.output
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(256, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(len(train_counter.keys()), activation='softmax')(head_model)

model = Model(inputs=conv_base.input, outputs=head_model)

for layer in conv_base.layers:
  layer.trainable = False

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=6)

batch_index = 0
data_list = []
while batch_index <= validation_imageflow.batch_index:
  data = validation_imageflow.next()
  data_list.append(data[0])
  batch_index += 1

print('head_model training started')
history = model.fit(
    x=train_imageflow,
    epochs=epochs,
    validation_data=validation_imageflow,
    callbacks=[early_stopping]
    )

train_imageflow.reset()
validation_imageflow.reset()

for layer in conv_base.layers[-12:]:
  layer.trainable = True

for n, layer in enumerate(conv_base.layers):
  if layer.trainable:
    print(f'{n+1}. of {len(conv_base.layers)}: {layer} is trainable = {layer.trainable}')

print('Model recompilation')
opt = Adam(learning_rate=0.0001)
accuracy_callback = AccuracyCallback(validation_imageflow)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print('unfreezed model training started')
history = model.fit(
    x=train_imageflow,
    epochs=epochs,
    validation_data=validation_imageflow,
    callbacks=[early_stopping, accuracy_callback]
    )

# 5. vypis vysledku a vizualizace
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
xepochs = range(1, len(accuracy)+1)

plt.plot(xepochs, accuracy, 'bo', label='Training accuracy')
plt.plot(xepochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Accuracy')
plt.figure()

plt.plot(xepochs, loss, 'bo', label='Training loss')
plt.plot(xepochs, val_loss, 'b', label='Validation loss')
plt.title('Loss')
plt.figure()

df = pd.DataFrame(history.history)
print(df)

plt.show()