import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Series s imenima
s1 = pd.Series(['crvenkapica', 'baka', 'majka', 'lovac', 'vuk'])
print(s1)

# Series s istom vrijednosti 5.0 za svaki indeks
s2 = pd.Series(5.0, index=['a', 'b', 'c', 'd', 'e'], name='ime_objekta')
print(s2)

# Series s 5 slučajnih brojeva
s3 = pd.Series(np.random.randn(5))
print(s3)


data = {
    'country': ['Italy', 'Spain', 'Greece', 'France', 'Portugal'],
    'population': [59, 47, 11, 68, 10],
    'code': [39, 34, 30, 33, 351]
} #riječnik

countries = pd.DataFrame(data, columns=['country', 'population', 'code'])
print(countries)

dataCSV = pd.read_csv("data_C02_emission.csv")

print(len(dataCSV))
print(dataCSV)

print(dataCSV.head(10))
print(dataCSV.tail(10))

print(dataCSV.info())

for col in ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']:
    dataCSV[col] = dataCSV[col].astype('category') #pretvorba object tipa u category

print(dataCSV.info())    

print(dataCSV.Make)
print(dataCSV[['Make', 'Model']])

print(dataCSV.iloc[2:6, 2:7]) #redovi 2-5, stupci 2-6
print(dataCSV.iloc[:, 2:5])
print(dataCSV.iloc[:, [0, 1, 2]])

print(dataCSV[dataCSV.Make=='Audi']) #samo vrijednosti koje su marke Audi, logički uvjeti na stupcima

dataCSV['large'] = (dataCSV['Cylinders'] > 10) #stvara novi stupac s vijednostima koje odgovaraju logičkom uvjetu

print(dataCSV)

new_data = dataCSV.groupby('Cylinders')
print(new_data.count()) #broji redove u svakoj grupi bez NaN
print(new_data.size()) #broji redove s NaN
# print(new_data.sum()) sumira numeričke vrijednosti u stupcima


print(dataCSV.isnull().sum())

dataCSV.dropna(axis=0) #brisanje redova, axis=0 -> obradi po redovima, briše redove koji imaju barem jedan NaN
dataCSV.dropna(axis=1) #axis=1 -> obradi po stupcima, briše sve stupce koji imaju barem jedan NaN
# ne mijenju originalni DataFrame, osim ako se ne doda inplace=True -> (axis=0, inplace=True)

dataCSV.drop_duplicates() #brisanje dupliciranih redova
#nakon brisanja redova, treba resetirati indekse retka
dataCSV = dataCSV.reset_index(drop=True)

#VIZUALIZACIJA PODATAKA

plt.figure()
dataCSV['Fuel Consumption City (L/100km)'].plot(kind='hist', bins=20)

plt.figure()
dataCSV['Fuel Consumption City (L/100km)'].plot(kind='box')
plt.show()


grouped = dataCSV.groupby('Cylinders')
grouped.boxplot(column=['CO2 Emissions (g/km)'])

dataCSV.boxplot(column=['CO2 Emissions (g/km)'], by='Cylinders')
plt.show()

dataCSV.plot.scatter(
    x='Fuel Consumption City (L/100km)',
    y='Fuel Consumption Hwy (L/100km)',
    c='Engine Size (L)',
    cmap="hot",
    s=50
)
plt.show() #za međusobni odnos dviju veličina se koristi dijagram raspršenja, ako se želi uključiti
# treća veličina, može se preko boje ili veličine pojedine točke

print(dataCSV.corr(numeric_only=True)) # korelacija mjeri koliko snažno i u kojem smjeru su povezane dvije numeričke varijable (brojčane vrijednosti)
#vrijednost ide od -1 do +1