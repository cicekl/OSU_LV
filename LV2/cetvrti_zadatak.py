import numpy as np
import matplotlib.pyplot as plt

black = np.zeros((50,50))
white = np.ones((50,50))

a = np.hstack([black,white]) #horizontalno povezivanje
b = np.hstack([white,black])

c = np.vstack([a,b]) #vertikalno povezivanje
plt.figure()
plt.imshow(c, cmap='gray') #obavezno cmap za sive tonove!!
plt.show()