# Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
# ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja data pri cemu je u 
# prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci
# stupac polja je masa u kg.

import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("data.csv", delimiter=",", skiprows=1) #preskače prvu liniju ako sadrži opis stupaca, delimiter dijeli riječi zarezom


# a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja? 
print(data.size)

# b) Prikažite odnos visine i mase osobe pomocu naredbe matplotlib.pyplot.scatter.
height = data[:,1]
weight = data[:,2]
plt.scatter(height, weight, c='r', s=1) #s je veličina točaka
plt.xlabel('visina')
plt.ylabel('masa')
plt.show()


# c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
height50 = height[::50] #uzima svaku 50. osobu, [start:stop:step]
weight50 = weight[::50]
plt.scatter(height50, weight50, c='r', s=1) #s je veličina točaka
plt.xlabel('visina')
plt.ylabel('masa')
plt.show()

# d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom 
# podatkovnom skupu.
print(np.min(height))
print(np.max(height))
print(np.mean(height))

# e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
# muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
# ind = (data[:,0] == 1)

ind = (data[:,0] == 1)
men_data = data[ind]

ind_w = (data[:,0] == 0) # provjera jesu li žene
women_data = data[ind_w]

print(np.min(men_data[:,1])) #uzmi visinu na indeksu 1
print(np.max(men_data[:,1]))
print(np.mean(men_data[:,1]))
print(np.min(women_data[:,1]))
print(np.max(women_data[:,1]))
print(np.mean(women_data[:,1]))