import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report


# Skripta zadatak_2.py učitava podatkovni skup Palmer Penguins [1]. 
# Ovaj podatkovni skup sadrži mjerenja provedena na tri različite vrste pingvina (Adelie, Chinstrap, Gentoo) 
# na tri različita otoka u području Palmer Station, Antarktika.

# Vrsta pingvina odabrana je kao izlazna veličina, pri čemu su klase označene 
# cjelobrojnim vrijednostima 0, 1 i 2. Ulazne veličine su duljina kljuna (bill_length_mm) i 
# duljina peraje u mm (flipper_length_mm).

# Za vizualizaciju podatkovnih primjera i granice odluke u skripti 
# je dostupna funkcija plot_decision_region.

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


# a) Pomocu stupcastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu 
# pingvina) u skupu podataka za ucenje i skupu podataka za testiranje. Koristite numpy 
# funkciju unique.
train_classes, train_counts = np.unique(y_train, return_counts = True)
plt.bar(train_classes, train_counts, tick_label = ['Adelie', 'Chinstrap', 'Gentoo'])
plt.title('Number of examples per class (train)')
plt.ylabel('Number of examples')
plt.show()

test_classes, test_counts = np.unique(y_test, return_counts = True)
plt.bar(test_classes, test_counts, tick_label = ['Adelie', 'Chinstrap', 'Gentoo'])
plt.title('Number of examples per class (test)')
plt.ylabel('Number of examples')
plt.show()

# b) Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa poda- 
# taka za ucenje. 
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# c) Pronadite u atributima izgradenog modela parametre modela. Koja je razlika u odnosu na ¯
# binarni klasifikacijski problem iz prvog zadatka?
print('Parameters: ', LogRegression_model.coef_)
#Razlika u odnosu na binarni klasifikacijski problem iz prvog zadatka je u broju klasa. U prvom zadatku imamo samo dvije klase 
# (binarni problem), dok u ovom zadatku imamo tri klase.

# d) Pozovite funkciju plot_decision_region pri cemu joj predajte podatke za ucenje i 
# izgradeni model logisticke regresije. Kako komentirate dobivene rezultate? 
plot_decision_regions(X_train, y_train, classifier = LogRegression_model)
#funkcija ne radi i baca grešku kod pozivanja


# e) Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke 
# regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izracunajte tocnost. 
# Pomocu classification_report funkcije izracunajte vrijednost  cetiri glavne metrike na skupu
# podataka za testiranje
y_test_p = LogRegression_model.predict(X_test) #KLASIFIKACIJA SKUPA PODATAKA!!
cm = confusion_matrix(y_test, y_test_p)
print('Matrica zabune: ', cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print('Accuracy: ', accuracy_score(y_test, y_test_p))
print('Classification report: ', classification_report(y_test, y_test_p))
