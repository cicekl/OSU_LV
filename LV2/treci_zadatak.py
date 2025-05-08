# Zadatak 2.4.3 Skripta zadatak_3.py ucitava sliku ’road.jpg’. Manipulacijom odgovarajuce
# numpy matrice pokušajte:
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")

# a) posvijetliti sliku,
brightness = 50
brightened_image = np.clip(img.astype(np.uint16) + brightness, 0, 255).astype(np.uint8)
plt.figure()
plt.imshow(brightened_image)
plt.show()

# b) prikazati samo drugu cetvrtinu slike po širini, 
width = img.shape[1]
second_quarter = width // 2
second_quarter_img = img[:, second_quarter:]
plt.figure()
plt.imshow(second_quarter_img)
plt.show()

# c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
rotated_img = np.rot90(img, k = 1)
plt.figure()
plt.imshow(rotated_img)
plt.show()

# d) zrcaliti sliku.
mirrored_img = np.flip(img, axis = 1)
plt.figure()
plt.imshow(mirrored_img)
plt.show()