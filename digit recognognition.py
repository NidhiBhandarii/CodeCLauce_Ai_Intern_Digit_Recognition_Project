import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import cv2

mnist = tf.keras.datasets.mnist
#x being the pixel data(digits)  and y classification.
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

#Normlizing
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)


# CNN model with dropout and batch normalization layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])

# Train the model
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs= 20)

model.save('Handwritten_model.keras')


model = tf.keras.models.load_model("Handwritten_model.keras")

loss,accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

# Load and test model with custom images
img_num = 1
while os.path.isfile(f"digits/digit{img_num}.png"):
    try:
        img1 = cv2.imread(f"digits/digit{img_num}.png", cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (28, 28))
        img1 = np.invert(img1)
        img1 = img1.reshape(1, 28, 28, 1)
        img1 = img1 / 255.0
        prediction = model.predict(img1)
        print(f"The digit is probably a {np.argmax(prediction)}")
        plt.imshow(img1.reshape(28, 28), cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"ERROR!!: {e}")
    finally:
        img_num += 1
