import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from keras.models import load_model
import keras.utils as image


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
for i in range(6):
    plt.imshow(x_train[i], cmap='gray')
    print('Oznaka slike: ', y_train[i])
    plt.show()


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape = (28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation = 'relu'))
model.add(layers.Dense(50, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))
model.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"],)


# TODO: provedi ucenje mreze
batch_size = 32
epochs = 20
history = model.fit(x_train_s,
                    y_train_s,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_split = 0.1)
predictions = model.predict(x_test_s)
y_test_p = np.argmax(predictions, axis = 1)


# TODO: Prikazi test accuracy i matricu zabune
score = model.evaluate(x_test_s,y_test_s,verbose=0)
print('Accuracy: ', score[1])
cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.show()


# TODO: spremi model
model.save('model.keras')
del model


#ZADATAK 8.4.2. Napišite skriptu koja ce ucitati izgradenu mrežu iz zadatka 1 i MNIST skup 
# podataka. Pomocu matplotlib biblioteke potrebno je prikazati nekoliko loše klasificiranih slika iz
# skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predvidenu

model = load_model('model.keras')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

predictions = model.predict(x_test)
y_test_p = np.argmax(predictions, axis = 1)

for i in range(100):
    if(y_test[i] != y_test_p[i]):
        plt.imshow(x_test[i])
        plt.title(f'Stvarna oznaka: {y_test[i]}, predvidena oznaka: {y_test_p[i]}')
        plt.show()


#ZADATAK 8.2.3. Napišite skriptu koja ce u citati izgradenu mrežu iz zadatka 1. Nadalje, skripta 
# treba ucitati sliku  test.png sa diska. Dodajte u skriptu kod koji ce prilagoditi sliku za mrežu, 
# klasificirati sliku pomocu izgradene mreže te ispisati rezultat u terminal. Promijenite sliku 
# pomocu nekog grafickog alata (npr. pomocu Windows Paint-a nacrtajte broj 2) i ponovo pokrenite 
# skriptu. Komentirajte dobivene rezultate za razlicite napisane znamenke.

model = load_model('model.keras')

img = image.load_img('test1.png', target_size = (28, 28), color_mode = 'grayscale')
img_array = image.img_to_array(img)

img_array = img_array.astype('float32') / 255
img_array_s = np.expand_dims(img_array, -1)

img_array_ready = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array_ready)

plt.imshow(img_array, cmap='gray')
plt.title(f'Predviđena oznaka: {prediction.argmax()}')
plt.show()