# algoritam K najbližih susjeda: 
# class sklearn.neighbors.KNeighborsClassifier( n_neighbors =5 , *,
# weights =’ uniform ’, algorithm =’ auto ’, leaf_size =30 , p=2 ,
# metric =’ minkowski ’, metric_params = None , n_jobs = None )

#n_neighbours - broj susjeda
# p - potencija na Minkowski metriku udaljenosti

#metode:
# .fit(X,y) - memorira sve podatkovne primjere za učenje 
# .predict(X) - izračunavanje izlaza modela na temelju ulaznih vrijednosti ulaznih veličina 
# .predict_proba(X) - izračunavanje vjerojatnosti klase na temelju ulaznih vrijednosti ulaznih veličina
# .kneighbours - izračunavanje indeksa i udaljenost najbližih susjeda za primjere u X



#SVM algoritam
#class sklearn.svm.SVC(*, C=1 .0 , kernel =’ rbf ’, degree =3 ,
#gamma =’ scale ’, coef0 =0 .0 , shrinking = True , probability = False ,
#tol =0. 001 , cache_size =200 , class_weight = None , verbose = False ,
#max_iter = -1 , decision_function_shape =’ ovr ’,
#break_ties = False , random_state = None )

#C - regularizacijski parametar
# kernel - definicija kernel funkcije koja se koristi
# gamma - kernel koeficijent za kernele 

#metode: 
# .fit(X,y) - za procjenu parametara modela na temelju podataka za učenje
# .predict(X) - za izračunavanje izlaza modela na temelju ulaznih vrijednosti ulaznih veličina
# .predict_proba(X) - za izračunavanje vjerojatnosti klase na temelju ulaznih vrijednosti 
# ulaznih veličina

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

#inicijalizacija i učenje KNN modela
KNN_model = KNeighborsClassifier(n_neighbors=5)
# KNN_model.fit(X_train_s, y_train) #skalirani podatci

#inicijalizacija i učenje SVM modela 
SVM_model = svm.SVC(kernel='rbf', gamma=1, C=0.1)
# SVM_model.fit(X_train_s, y_train)

#predikcija na skupu podataka za testiranje
# y_test_p_KNN = KNN_model.predict(X_test)
# y_test_p_SVM = SVM_model.predict(X_test)



#METODA UNAKRSNE VALIDACIJE:
# sklearn.model_selection.cross_val_score( estimator , X , y = None ,
# *, groups = None , scoring = None , cv= None , n_jobs = None , verbose =0 ,
# fit_params = None , pre_dispatch =’2 * n_jobs ’, error_score = nan

# estimator - scikit-learn model
# X - vrijednosti ulaznih veličina
# y - vrijednosti izlaznih veličina
# scoring – string koji definira metriku za evaluaciju modela ako se ne koristi scoring modela
# cv – cjelobrojna vrijednost koja definira broj podskupova na koji se dijeli skup za učenje
# funkcija vraća: scores – polje koje sadrži rezultate za svaku iteraaciju unakrsne validacije

from sklearn.model_selection import cross_val_score

model =svm.SVC(kernel='linear', C=1 , random_state =42 )
# scores = cross_val_score ( clf , X_train , y_train , cv =5 )
# print ( scores )

# k-struka unakrsna validacija se koristi kako bi se pronašla optimalna vrijednost
# hiperparametra određene metode
# često metode imaju više od jednog hiperparametra pa se procjena
# optimalnih vrijednosti hiperparametara radi „rešetkastom“ pretragom