import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

data = pd.read_csv('data_C02_emission.csv')


# Skripta zadatak_1.py ucitava podatkovni skup iz data_C02_emission.csv.
# Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih velicina. Detalje oko ovog podatkovnog skupa mogu se pronaci u 3. 
# laboratorijskoj vježbi.

# a) Odaberite željene numericke velicine specificiranjem liste s nazivima stupaca. Podijelite
# podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%.

print(data.info())

X = data[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)']]
y = data['CO2 Emissions (g/km)'] #gradimo model koji procjenjuje emisiju C02 plinova

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1) #random state omogucuje jednaku/istu podjelu, podjela 80%-20%


# b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova 
# o jednoj numerickoj velicini. Pri tome podatke koji pripadaju skupu za ucenje oznacite 
# plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.
plt.scatter(X_train['Engine Size (L)'], y_train, color = 'blue', label = 'Training data')
plt.scatter(X_test['Engine Size (L)'], y_test, color = 'red', label = 'Testing data')

plt.show()

# c) Izvršite standardizaciju ulaznih velicina skupa za ucenje. Prikažite histogram vrijednosti 
# jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja 
# transformirajte ulazne velicine skupa podataka za testiranje.

plt.hist(X_train['Cylinders'], color = 'green')
plt.show() #prikaz prije skaliranja ulaznih podataka

sc = MinMaxScaler()
X_train_s = sc.fit_transform(X_train) #skaliranje ulaznih podataka, "uči" i transformira
X_train_s = pd.DataFrame(X_train_s, columns = X_train.columns) #rezultat je NumPy niz pa ga pretvaramo u DataFrame
X_test_s = sc.transform(X_test) #transformacija test skupa
X_test_s = pd.DataFrame(X_test_s, columns = X_test.columns) #rezultat je NumPy niz pa ga pretvaramo u DataFrame

plt.hist(X_train_s['Cylinders'], color = 'red') #prikaz poslije skaliranja ulaznih podataka
plt.show()

# d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
# povežite ih s izrazom 4.6.
linearModel = lm.LinearRegression()
linearModel.fit(X_train_s, y_train) #treniranje modela na skaliranim značajkama (X_train_s) i ciljevima (y_train)
print("θ₀ (intercept):", linearModel.intercept_) #ispisuje presjek
for name, coef in zip(X_train.columns, linearModel.coef_): #ispis svakog koeficijenta, zip spaja imena i koeficijente
    print(f"θ za {name}: {coef:.4f}")

# e) Izvršite procjenu izlazne velicine na temelju ulaznih velicina skupa za testiranje. Prikažite 
# pomocu dijagrama raspršenja odnos izmedu stvarnih vrijednosti izlazne velicine i procjene 
# dobivene modelom.

y_test_p = linearModel.predict(X_test_s)
plt.scatter(X_test_s['Fuel Consumption City (L/100km)'], y_test, color = 'blue', label = 'Real values')
plt.scatter(X_test_s['Fuel Consumption City (L/100km)'], y_test_p, color = 'red', label = 'Predicted values')
plt.show()

# f) Izvršite vrednovanje modela na nacin da izracunate vrijednosti regresijskih metrika na 
 # skupu podataka za testiranje.
MAE = mean_absolute_error(y_test, y_test_p)
print('Mean absolute error: ', MAE)
MSE = mean_squared_error(y_test, y_test_p)
print('Mean squared error: ', MSE)
RMSE = math.sqrt(MSE)
print('Root mean squared error: ', RMSE)
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
print('Mean absolute percentage error: ', MAPE)
R2 = r2_score(y_test, y_test_p)
print('R2 score: ', R2)

 # Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj 
# ulaznih velicina?
# R2 raste, a MAE,MSE,RMSE,MAPE padaju

