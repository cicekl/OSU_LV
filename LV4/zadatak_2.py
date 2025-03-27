import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm

# Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku 
# varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategorickih 
# velicina. Radi jednostavnosti nemojte skalirati ulazne velicine. Komentirajte dobivene rezultate. 
# Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
# vozila radi?


data = pd.read_csv('data_C02_emission.csv')


ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()

numerical_features = data.select_dtypes(include='number')
ohe_columns = ohe.get_feature_names_out(['Fuel Type'])
X_encoded_df = pd.DataFrame(X_encoded, columns=ohe_columns, index=data.index)
numerical_features = pd.concat([numerical_features, X_encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(numerical_features.drop(['CO2 Emissions (g/km)'], axis=1), numerical_features['CO2 Emissions (g/km)'], test_size=0.2, random_state=1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)
print('Model coefficients: ', linearModel.coef_)



y_test_prediction = linearModel.predict(X_test)
plt.scatter(y_test, y_test_prediction)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.show()

absolute_errors = abs(y_test - y_test_prediction)

max_error_index = absolute_errors.idxmax()
max_error = absolute_errors[max_error_index]

vehicle_model = data.loc[max_error_index, 'Model']

print('Maximum absolute error: ', max_error)
print('Model of the vehicle associated with maximum error: ', vehicle_model)