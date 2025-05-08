from sklearn import datasets
from sklearn.metrics import mean_absolute_error
from sklearn . model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn . linear_model as lm

#linearno regresijski model - to je model koji pokušava naučiti linearnu vezu između ulaznih varijabli (X) 
# i ciljne varijable (y). Cilj je predvidjeti vrijednost y na temelju vrijednosti X.


# ucitaj ugradeni podatkovni skup
X , y = datasets.load_diabetes(return_X_y=True)
# podijeli skup na podatkovni skup za ucenje i poda tkovni skup za testiranje
X_train,X_test,y_train,y_test = train_test_split(X ,y ,test_size = 0.2 ,random_state=1)

sc = MinMaxScaler() #transformira podatke tako da svi brojevi budu u istom rasponu, obično [0, 1]
#napravi se scaler objekt
X_train_s = sc.fit_transform(X_train) #izračuna min i max iz X_train,odmah i skalira podatke
X_test_s = sc.transform(X_test) #koristi iste min i max vrijednosti iz X_train da normalizira X_test

ohe = OneHotEncoder() #kreira encoder koji će pretvoriti kategorijske vrijednosti u tzv. "one-hot" vektore.

 # X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray() -> uči koje su kategorije u stupcu te ih transformira u one-hot vektore

 # .fit(X, y) - > trenira (uči) model na skupu podataka X (ulazi) i y (ciljevi), model na temelju tih podataka procjenjuje parametre
 # .predict(X) -> koristi naučene parametre da predvidi rezultat za nove ulaze X

model = lm.LinearRegression()
model.fit(X_train_s, y_train)  # treniranje
y_test_p = model.predict(X_test_s)  #predikcija

# evaluacija modela na skupu podataka za testiranje pomocu MAE
MAE = mean_absolute_error( y_test,y_test_p)