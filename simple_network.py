from google.colab import drive
drive.mount('/content/drive')

!unzip -qq '__PATH_TO_GDRIVE_FOLDER__/cwe3categ.zip'
!unzip -qq '__PATH_TO_GDRIVE_FOLDER__/digits_small.zip'
!unzip -qq '__PATH_TO_GDRIVE_FOLDER__/cwe3categ_augmented.zip'

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend
from tensorflow.keras import metrics

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# reset all states
backend.clear_session()

image_shape = (330, 330, 3)
# zakomentovany kod -> rozpoznani cislic
#image_shape = (28, 28, 3)

# 1. definice modelu
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=image_shape))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
#model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
#model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(3, activation='softmax'))

# vypis vrstev modelu
model.summary()
# learning_rate otestovan od default 0.01 do 10^-6,
# nastaveni 0.0001 dava neljepsi vysledky
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 2. nacteni dat
# Zvyseni poctu trenovacich dat
train_imagedatagen = ImageDataGenerator(rotation_range=45,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              fill_mode='nearest')

validation_imagedatagen = ImageDataGenerator()

batch_size = 20
train_path = '/content/_splitted/train'
#train_path = '/content/digits_small/train'
train_path = '/content/cwe3categ_augmented/train'
train_imageflow = train_imagedatagen.flow_from_directory(train_path,
                                               color_mode= 'rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               target_size = image_shape[:2],)

test_path = '/content/_splitted/validation'
#test_path = '/content/digits_small/validation'
#test_path = '/content/cwe3categ_augmented/validation'
validation_imageflow = validation_imagedatagen.flow_from_directory(test_path,
                                               target_size = image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=False)

print(validation_imageflow.class_indices.keys())
print(validation_imageflow.class_indices)

epochs = 20
early_stopping = EarlyStopping(monitor='val_loss', patience=6)

# 3. sestaveni modelu a spusteni uceni,
history = model.fit(
    train_imageflow,
    epochs=epochs,
    validation_data=validation_imageflow,
    callbacks=[early_stopping],
    )

# 4. zobrazeni vysledku
Y_pred = model.predict(validation_imageflow)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_imageflow.classes, y_pred))
print('Classification Report')
target_names = list(validation_imageflow.class_indices.keys()) #['Bleeding', 'Normal', 'Polypoids'] 
print(classification_report(validation_imageflow.classes, y_pred, target_names=target_names))


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
xepochs = range(1, len(accuracy)+1)

# sestaveni grafu, 'bo' - blue dots, 'b' - blue line
plt.plot(xepochs, accuracy, 'bo', label='Training accuracy')
plt.plot(xepochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Accuracy')
plt.figure()

plt.plot(xepochs, loss, 'bo', label='Training loss')
plt.plot(xepochs, val_loss, 'b', label='Validation loss')
plt.title('Loss')
plt.figure()

# vystupy v history.history ulozim jako pandas dataframe
df = pd.DataFrame(history.history)
print(df)

# spusteni vykresleni
plt.show()