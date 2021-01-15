from google.colab import drive
drive.mount('/content/drive')

!unzip -qq '__PATH_TO_FOLDER__/cwe3categ.zip'
!unzip -qq '__PATH_TO_FOLDER__/digits_small.zip'
!unzip -qq '__PATH_TO_FOLDER__/cwe3categ_augmented.zip'

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback
from keras import backend
from keras.applications import VGG16

from keras.metrics import Precision, Recall
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

# Trida callbacku - pocita accuracy v ramci trid, je spustena na konci epoch
class AccuracyCallback(Callback):
    def __init__(self, test_data, classes):
        self.test_data = test_data
        self.classes = list(classes)

    def on_epoch_end(self, epoch, logs=None):
        x_data, y_data = self.test_data

        correct = 0
        incorrect = 0
        x_result = model.predict(x_data, verbose=0)
        class_correct = [0] * len(self.classes)
        class_incorrect = [0] * len(self.classes)

        for i in range(len(x_data)):
            x = x_data[i]
            y = y_data[i]
            res = x_result[i]

            actual_label = np.argmax(y)
            pred_label = np.argmax(res)

            if(pred_label == actual_label):
                class_correct[actual_label] += 1   
                correct += 1
            else:
                class_incorrect[actual_label] += 1
                incorrect += 1
        print('\tclass_correct =', class_correct, '=', correct)
        print('\tclass_incorrect =', class_incorrect, '=', incorrect)

        for i in range(len(self.classes)):
            tot = float(class_correct[i] + class_incorrect[i])
            class_acc = -1
            if (tot > 0):
                class_acc = float(class_correct[i]) / tot

        acc = float(correct) / float(correct + incorrect)  
        print(f'Current Network Accuracy: acc({acc}) = {correct} / {correct+incorrect}')

# funkce pro extrakci features z vlastni datove sady protazene pres VGG16
def extract_features_from_conv_layers(source_dir):
  imageflow = imagedatagen.flow_from_directory(
      source_dir, 
      color_mode= 'rgb', 
      batch_size=batch_size,
      class_mode='categorical',
      target_size = image_shape[:2])
  
  counter = Counter(imageflow.classes)
  print(counter.items())
  class_dict = imageflow.class_indices
  sample_count = imageflow.samples
  features = np.zeros(shape=(sample_count, 10, 10, 512))
  labels = np.zeros(shape=(sample_count,len(class_dict.keys())))

  i = 0
  for input_batch, label_batch in imageflow:
    features[i*batch_size : (i+1)*batch_size] = conv_base.predict(input_batch)
    labels[i*batch_size : (i+1)*batch_size] = label_batch
    i += 1
    if i*batch_size >= sample_count:
      print('sample processing finished: i*batch_size >= sample_count =', i*batch_size, '>=', sample_count)
      break

  return features, labels, sample_count, counter, imageflow

# reset all states
backend.clear_session()

image_shape = (330, 330, 3)
#image_shape = (32, 32, 3)

# konvolucni baze z VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)
for layer in conv_base.layers[:]:
  layer.trainable = False
# vypis vrstev modelu
conv_base.summary()

# 2. nacteni dat a jejich prepocet dle konvoluce z VGG16
imagedatagen = ImageDataGenerator()
batch_size = 20

train_path = '/content/_splitted/train'
#train_path = '/content/digits_small/train'
train_path = '/content/cwe3categ_augmented/train'
#train_path = '/content/it4n_balanced'
#train_path = '__PATH_TO_FOLDER__/cwe_dataset/train'

train_features, train_labels, sample_count, classes, train_flow = extract_features_from_conv_layers(train_path)
print('sample_count:', sample_count)
print(train_features.shape)
train_features = np.reshape(train_features, (sample_count, 10*10*512))
print(train_features.shape)

validation_path = '/content/_splitted/validation'
#validation_path = '/content/digits_small/validation'
validation_path = '/content/cwe3categ_augmented/validation'
#validation_path = '__PATH_TO_FOLDER__/cwe_dataset/validation'
val_features, val_labels, sample_count, classes, val_flow = extract_features_from_conv_layers(validation_path)
print(val_features.shape)
val_features = np.reshape(val_features, (sample_count, 10*10*512))
print(val_features.shape)

# 3. definice vlastniho klasifikatoru k napojeni na vystupni features z VGG16,
# stejny klasifikator, jako v simple_network.py
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=51200))
model.add(Dropout(0.5))
model.add(Dense(len(classes.keys()), activation='softmax'))

# 4. sestaveni modelu a spusteni trenovani
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=6)
accuracy_callback = AccuracyCallback((val_features, val_labels), classes.keys())

class_weight = {0: 1.0, 1: 1.0, 2: 1.0}

epochs = 20
history = model.fit(
    train_features, train_labels,
    epochs=epochs,
    validation_data=(val_features, val_labels),
    callbacks=[early_stopping, accuracy_callback],
    class_weight=class_weight
    )

# 5. vypis vysledku a vizualizace
Y_pred = model.predict(val_features)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(val_flow.classes, y_pred))
print('Classification Report')
target_names = list(val_flow.class_indices.keys())
print(classification_report(val_flow.classes, y_pred, target_names=target_names))

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