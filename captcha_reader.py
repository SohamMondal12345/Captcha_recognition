import os
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Check if captcha_images_v2 directory exists, otherwise create it
data_dir = 'captcha_images_v2'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Constants and Variables
img_shape = (50, 200, 1) # 50 height, 200 width and 1- greyscale image 
symbols = list(map(chr, range(97, 123))) + list(map(chr, range(48, 58)))  # a-z + 0-9
len_symbols = len(symbols)  # 36
len_captcha = 5

# Function to generate random captcha text
def generate_random_text(length=len_captcha):
    return ''.join(random.choices(symbols, k=length))

# Function to generate and save captcha images
def generate_captcha_images(num_images=1040):
    for i in range(num_images):
        captcha_text = generate_random_text()
        image = np.zeros((50, 200), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, captcha_text, (30, 35), font, 1.2, (255), 2)
        cv2.imwrite(os.path.join(data_dir, f'{captcha_text}.png'), image)

# Generate captcha images if the directory is empty
if len(os.listdir(data_dir)) == 0:
    generate_captcha_images()

# Load and preprocess data
nSamples = len(os.listdir(data_dir)) 
X = np.zeros((nSamples, *img_shape)) #array to hold captcha images
y = np.zeros((len_captcha, nSamples, len_symbols)) #array to hold captcha lables for traininig

# Iterate through each captcha image
for i, captcha in enumerate(os.listdir(data_dir)):
    captcha_code = captcha.split(".")[0]
    captcha_cv2 = cv2.imread(os.path.join(data_dir, captcha), cv2.IMREAD_GRAYSCALE)
    captcha_cv2 = captcha_cv2 / 255.0
    captcha_cv2 = captcha_cv2.reshape(img_shape)

    for j, symbol in enumerate(captcha_code):
        y[j, i, symbols.index(symbol)] = 1  # One-hot encode the labels

    X[i] = captcha_cv2

# Splitting data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y.transpose(1, 0, 2), test_size=0.2, random_state=42)

# Define the model architecture
captcha_input = Input(shape=img_shape, name='captcha_input')

x = Conv2D(16, (3, 3), padding='same', activation='relu')(captcha_input)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = BatchNormalization()(x)
x = Flatten()(x)

# Shared dense layers
dense1 = Dense(64, activation='relu')(x)
dropout1 = Dropout(0.5)(dense1)
output1 = Dense(len_symbols, activation='softmax', name='char_1')(dropout1)

dense2 = Dense(64, activation='relu')(x)
dropout2 = Dropout(0.5)(dense2)
output2 = Dense(len_symbols, activation='softmax', name='char_2')(dropout2)

dense3 = Dense(64, activation='relu')(x)
dropout3 = Dropout(0.5)(dense3)
output3 = Dense(len_symbols, activation='softmax', name='char_3')(dropout3)

dense4 = Dense(64, activation='relu')(x)
dropout4 = Dropout(0.5)(dense4)
output4 = Dense(len_symbols, activation='softmax', name='char_4')(dropout4)

dense5 = Dense(64, activation='relu')(x)
dropout5 = Dropout(0.5)(dense5)
output5 = Dense(len_symbols, activation='softmax', name='char_5')(dropout5)

# Model definition and compilation
model = keras.models.Model(inputs=captcha_input, outputs=[output1, output2, output3, output4, output5])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Training the model
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
history = model.fit(X_train, [y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3], y_train[:, 4]],
                    batch_size=32, epochs=50, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on test data
score = model.evaluate(X_test, [y_test[:, 0], y_test[:, 1], y_test[:, 2], y_test[:, 3], y_test[:, 4]], verbose=1)
print('Test Loss and Accuracy:', score)

# Save the trained model
model.save('captcha_model.h5')
