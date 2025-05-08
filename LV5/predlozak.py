from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib . pyplot as plt
from sklearn . metrics import accuracy_score
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay

#inicijalizacija i ucenje modela logisticke regresije
LogRegression_model = LogisticRegression()
# LogRegression_model.fit(X_train, y_train) -> procjena parametara modela na temelju podatka za učenje

#predikcija na skupu podataka za testiranje
# y_test_p = LogRegression_model.predict(X_test)-> izracunavanje izlaza modela na temelju ulaznih vrijednosti
# ulaznih velicina

# stvarna vrijednost izlazne velicine i predikcija
y_true = np.array ([1 , 1 , 1 , 0 , 1 , 0 , 1 , 0 , 1])
y_pred = np.array ([0 , 1 , 1 , 1 , 1 , 0 , 1 , 0 , 0])

# tocnost
print ("Tocnost:",accuracy_score(y_true, y_pred))

# matrica zabune
cm = confusion_matrix( y_true,y_pred)
print("Matrica zabune:",cm)
disp = ConfusionMatrixDisplay(confusion_matrix( y_true, y_pred))
disp.plot()
plt.show()

# report -> izačun četiri glavne metrike (točnost, preciznost, odziv i F1 mjeru)
# print(classification_report( y_true,y_pred))
