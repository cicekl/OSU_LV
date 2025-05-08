# Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv

import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

print(data)

# a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili 
# duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke velicine konvertirajte u tip 
# category.

print("Broj mjerenja:")
print(len(data))

print(data.info())

isTrue = data.isnull().values.any()
printText = 'Postoje izostale vrijednosti' if isTrue else 'Ne postoje izostale vrijednosti'
print(printText)

isDuplicated = data.duplicated().values.any()
printText = 'Postoje duplicirane vrijednosti' if isDuplicated else 'Ne postoje duplicirane vrijednosti'
print(printText)

categories = data.select_dtypes(include='object').columns
print(categories)

for ctg in categories:
    data[ctg] = data[ctg].astype('category')

print(data.dtypes)

# b) Koja tri automobila imaju najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: 
# ime proizvodaca, model vozila i kolika je gradska potrošnja

sortedData = data.sort_values(by='Fuel Consumption City (L/100km)', ascending=True)
print('Najmanja gradska potrošnja:')
print(sortedData[['Make','Model','Fuel Consumption City (L/100km)']].head(3))
print('Najveća gradska potrošnja:')
print(sortedData[['Make','Model','Fuel Consumption City (L/100km)']].tail(3))


# c) Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? Kolika je prosjecna C02 emisija 
# plinova za ova vozila?

filtered = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print('Broj vozila s veličinom motora između 2.5 i 3.5:',len(filtered))

print('Prosječna CO2 emisija:')
print(filtered['CO2 Emissions (g/km)'].mean())

# d) Koliko mjerenja se odnosi na vozila proizvodaca Audi? Kolika je prosjecna emisija C02 
# plinova automobila proizvodaca Audi koji imaju 4 cilindara?

carsAudi = data[(data['Make'] == 'Audi')]
print('Vozila proizvođača Audi:')
print(len(carsAudi))

filteredAudi = carsAudi[(carsAudi['Cylinders'] == 4)]
print('Prosječna emisija C02 automobila Audi s 4 cilindra:')
print(filteredAudi['CO2 Emissions (g/km)'].mean())

# e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na 
# broj cilindara?

filteredCylinders = data[(data['Cylinders'] % 2 == 0) & (data['Cylinders'] >= 4)]
print('Broj vozila s cilindrima 4,6,8..:')
print(len(filteredCylinders))

avgEmissions = filteredCylinders.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()

print('Prosječna CO2 emisija po broju cilindara:')
print(avgEmissions)

# f) Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, a kolika za vozila 
# koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?

avgConsumption = data[(data['Fuel Type'] == 'D')]
print('Prosječna potrošnja goriva automobila koji koriste dizel:')
print(avgConsumption['Fuel Consumption City (L/100km)'].mean())
print(avgConsumption['Fuel Consumption City (L/100km)'].median())


avgConsumption = data[(data['Fuel Type'] == 'X')]
print('Prosječna potrošnja goriva automobila koji koriste regularni benzin:')
print(avgConsumption['Fuel Consumption City (L/100km)'].mean())
print(avgConsumption['Fuel Consumption City (L/100km)'].median())

# g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?
filteredCar = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
print('Vozilo s najvećom gradskom potrošnjom:')
print(filteredCar[filteredCar['Fuel Consumption City (L/100km)'] == filteredCar['Fuel Consumption City (L/100km)'].max()])

# h) Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)? 

filteredCars = data[(data['Transmission'].str.startswith('M'))]
print(len(filteredCars))

# i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat.

correlation = data.corr(numeric_only=True)
print('Korelacija:')
print(correlation)
# veći motor → više cilindara, veći motor → veća potrošnja u gradu, veći motor → više CO₂ emisija
# što više goriva vozilo troši → više CO₂ ispušta
# više mpg = manja potrošnja
# vozila koja više troše → više zagađuju
# veći motori su češće označeni kao "large" i više troše/emituju
