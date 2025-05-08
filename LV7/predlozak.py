# class sklearn.cluster.KMeans( n_clusters =8 , *, init =’k - means ++ ’,
# n_init =’ warn ’, max_iter =300 , tol =0 . 0001 , verbose =0 ,
# random_state = None , copy_x = True , algorithm =’ lloyd ’)

# n_clusters - broj grupa
# init - načini inicijalizacije centara
# n_init - koliko puta će se algoritam izvršiti s razl. početnim centrima
# max_iter - maksimalni broj iteracija algoritma

#metode:
# .fit(X) - izvršava algoritam K srednjih vrijednosti
# .predict(X) - izračunaj najbliži centar za svaki podatak u X

from sklearn.cluster import KMeans
import numpy as np

# podatkovni primjeri 
X = np.array([ [9 , 1], [3 , 2], [3 , 9], [4 , 8], [8 , 2],
[7 , 4], [9 , 7], [1 , 4], [8 , 7], [1 , 1]])

# inicijalizacija algoritma K srednjih vrijednosti
km=KMeans(n_clusters=3, init='random', n_init=5, random_state=0)

# pokretanje grupiranja primjera
km.fit(X)

# dodijeljivanje grupe svakom primjeru
labels = km.predict(X)