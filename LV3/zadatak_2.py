import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('data_C02_emission.csv')


# a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20)
plt.show()

# b) Pomocu dijagrama raspršenja prikažite odnos izmedu gradske potrošnje goriva i emisije 
# C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu
# velicina, obojite tockice na dijagramu raspršenja s obzirom na tip goriva

fuel_map = {'Z': 0, 'D': 1, 'E': 2, 'X': 3}
data['FuelCode'] = data['Fuel Type'].map(fuel_map) #mapiranje kategoričkih vrijednostti u numberičke 



plt.figure()
data.plot.scatter(x='Fuel Consumption City (L/100km)',
                  y='CO2 Emissions (g/km)',
                  color='blue')

data.plot.scatter(x='Fuel Consumption City (L/100km)',
                  y='CO2 Emissions (g/km)',
                   c=data['FuelCode'],
                  cmap='hot')

plt.show()


# c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip 
# goriva. Primjecujete li grubu mjernu pogrešku u podacima?

plt.figure()
data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.show()

# d) Pomocu stupcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu 
# groupby.

plt.figure()
grouped = data.groupby('Fuel Type').size()
grouped.plot(kind='bar', color='skyblue', edgecolor='black')
plt.show()

# e) Pomocu stupcastog grafa prikažite na istoj slici prosjecnu C02 emisiju vozila s obzirom na
# broj cilindara.
avgEmission = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
avgEmission.plot(kind='bar',color='salmon', edgecolor='black')
plt.show()
