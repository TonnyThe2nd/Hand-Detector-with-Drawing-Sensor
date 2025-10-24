import tensorflow as tf
import cv2
import numpy as np
modelo = tf.keras.models.load_model('modelo.h5')

imagem = tf.keras.utils.load_img('./train/C/7.png', target_size=(128,128))

img_gen = np.array(imagem)/255
img_gen = np.expand_dims(img_gen, axis = 0)


teste = modelo.predict(img_gen)
print(teste)
classe = np.argmax(teste)
print("resultado:" , classe)
print(modelo.weights)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
