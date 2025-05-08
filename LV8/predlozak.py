from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequnetial()
model.add(layers.Input(shape=3)) #definira se ulaz s 3 značajke
model.add(layers.Dense(3, activation = "relu")) #sloj s 3 neurona
model.add(layers.Dense(1, activation = "sigmoid")) #sloj s 1 neuronom

model.summary()

#konfiguracija modela za proces učenja 
model.compile( loss =" categorical_crossentropy " ,
optimizer =" adam ",
metrics = [" accuracy " ,])
batch_size = 32
epochs = 20
history = model . fit ( X_train ,
y_train ,
batch_size = batch_size ,
epochs = epochs ,
validation_split = 0 . 1)
predictions = model . predict ( X_test )
score = model . evaluate ( X_test , y_test , verbose =0 )