import matplotlib.pyplot as plt
import numpy as np

frozen_16_angolo = np.array([1,3,6,10,13,15,16]) / 16
frozen_64_angolo = np.array([1,3,6,10,15,21,28,36,43,49,54,58,61,63,64]) / 64
#taxi_centro = np.array([1,5,13,21,25]) / 25
#taxi_angolo = np.array([1,3,6,9,13,18,22,24,25]) / 25
taxi_centro_e_angolo = np.array([1,5,13,21,25,26,28,31,34,38,43,47,49,50]) / 50
blackjack = (np.array([5, 10, 15, 20]) * 2) / 64

mountain_car_pessimista = np.array([1,3,5,8,11,15,20,26,33,41,50,60,71,83,96,110,125]) / 285
mountain_car_realista = np.array([1,3,5,8,13,19,27,36,47,59,73,88,105,123,143,164,187]) / 285
mountain_car_ottimista = np.array([1,3,5,8,14,23,35,50,68,89,113,140,170,203,239,278,285]) / 285

cliffwalking_angolo = np.array([1,3,5,8,12,16,20,24,28,32,36,40,44,47,48]) / 48






plt.plot(frozen_16_angolo, label="frozen_16_angolo")
plt.plot(frozen_64_angolo, label="frozen_64_angolo")
#plt.plot(taxi_angolo, label="taxi_angolo")
#plt.plot(taxi_centro, label="taxi_centro")
plt.plot(taxi_centro_e_angolo, label="taxi_centro_e_angolo")
plt.plot(blackjack, label="blackjack")
plt.plot(mountain_car_pessimista, label="mountain_car_pessimista")
plt.plot(mountain_car_realista, label="mountain_car_realista")
plt.plot(mountain_car_ottimista, label="mountain_car_ottimista")
plt.plot(cliffwalking_angolo, label="cliffwalking_angolo")




plt.legend(loc='upper left')
plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
plt.xlabel("Numero di mosse")

plt.show()
