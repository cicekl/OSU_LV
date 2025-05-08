import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,6, num=30)
y = np.sin(x)
plt.plot(x,y,'r', linewidth =1, marker =".", markersize =5)
plt.axis([0,6,-2,2]) #raspon osi grafa
plt.xlabel('x')
plt.ylabel('vrijednosti funkcije')
plt.title('Sinus funkcija')
plt.show()

img = plt.imread("road.jpg") #slika ima oblik (visina, širina, 3) - 3 kanala ako je RGB
img = img[:,:,0].copy() #uzima se samo prvi kanal i pravi se kopija, za sive tonove
print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img, cmap="gray") #obavezno cmap za prikaz u sivim tonovima
plt.show()
