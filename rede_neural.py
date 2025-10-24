import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

PATH_TRAINING = 'train'
PATH_TESTING = 'test'

image_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split = 0.2,
    width_shift_range = 0.2,
    height_shift_range=0.2,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, validation_split=0.2)

training_images = image_data_gen.flow_from_directory(
    PATH_TRAINING,
    target_size=(128,128),
    shuffle=True,
    seed=10,
    class_mode='categorical',
    batch_size=64,
    subset='training'
)

validation_images = val_data_gen.flow_from_directory(
    PATH_TRAINING,
    target_size=(128,128),
    shuffle=False,
    seed=10,
    class_mode='categorical',
    batch_size=64,
    subset='validation'
)

test_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_images = test_image_gen.flow_from_directory(
    PATH_TESTING,
    target_size=(128,128),
    shuffle=False,
    seed=10,
    class_mode='categorical',
    batch_size=64
)

print(str(list(training_images.class_indices)))

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu',input_shape=(128,128,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(21, activation='softmax'))

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 50

model_check = tf.keras.callbacks.ModelCheckpoint(filepath ='modelo.h5', monitor='val_loss', save_best_only=True, verbose=1)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30, verbose=1, restore_best_weights=True,)

hist = model.fit(training_images, epochs = epochs, callbacks=[model_check, earlystop], verbose=1, validation_data = validation_images)

score = model.evaluate(validation_images)
print("Loss: ", score[0])

print("Acc: , ", score[1])