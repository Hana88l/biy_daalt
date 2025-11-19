# import cv2
# from pathlib import Path

# cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
# # zasvar
# clf = cv2.CascadeClassifier(str(cascade_path))

# camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

# while True:
#     ret, frame = camera.read()
#     if not ret:
#         print("Failed to access camera!")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = clf.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

#     cv2.imshow("Live Face Detection - Press Q to Quit", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()
# mnist_cnn.py
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, models

# # 1. Датасет унших
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# # хэвийн болгох ба dimension нэмэх
# x_train = x_train.astype('float32') / 255.0
# x_test  = x_test.astype('float32') / 255.0
# x_train = np.expand_dims(x_train, -1)  # (batch,28,28,1)
# x_test  = np.expand_dims(x_test, -1)

# # 2. Модель
# model = models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
#     layers.MaxPooling2D((2,2)),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D((2,2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.4),
#     layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # 3. Сургах
# history = model.fit(x_train, y_train, epochs=6, batch_size=128,
#                     validation_split=0.1)

# # 4. Үнэлэх
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f"Test accuracy: {test_acc:.4f}")

# # 5. Жишээ зураг дээр турших
# import cv2
# def predict_from_image(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (28,28))
#     img = 255 - img  # хэрэв фон цагаан биш бол invert хэрэгтэй байж болно
#     img = img.astype('float32') / 255.0
#     img = img.reshape(1,28,28,1)
#     pred = model.predict(img)
#     return np.argmax(pred), np.max(pred)

# # Туршилт
# print(predict_from_image('digit_photo.png'))


import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n Test Accuracy: {test_acc*100:.2f}%")

predictions = model.predict(x_test)

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.title(np.argmax(predictions[i]))
plt.show()