import numpy as np
import matplotlib.pyplot as plt


#Taxi
taxi_x = [13]
taxi_y = [1.0]
plt.scatter(taxi_x, taxi_y, label="Taxi")

#Taxi (MOD1)
taxi_x = [13]
taxi_y = [0.33]
plt.scatter(taxi_x, taxi_y, label="Taxi (MOD1)")

#Roulette
roulette_x = [0]
roulette_y = [0.97]
plt.scatter(roulette_x, roulette_y, label="Roulette")

#MountainCar
mountainCar_x = [20]
mountainCar_y = [0.861]
plt.scatter(mountainCar_x, mountainCar_y, label="MountainCar")

#Blackjack
blackjack_x = [4]
blackjack_y = [0.5]
plt.scatter(blackjack_x, blackjack_y, label="Blackjack")

#FrozenLake4x4
frozenLake4x4_x = [4]
frozenLake4x4_y = [0.033]
plt.scatter(frozenLake4x4_x, frozenLake4x4_y, label="FrozenLake4x4")

#FrozenLake8x8
frozenLake8x8_x = [14]
frozenLake8x8_y = [0.0079]
plt.scatter(frozenLake8x8_x, frozenLake8x8_y, label="FrozenLake8x8")

#Cliff walking
cliff_walking_x = [14]
cliff_walking_y = [1.0]
plt.scatter(cliff_walking_x, cliff_walking_y, label="Cliff walking")

#Cliff walking (MOD1)
cliff_walking_x = [14]
cliff_walking_y = [0.07]
plt.scatter(cliff_walking_x, cliff_walking_y, label="Cliff walking (MOD1)")



plt.gca().invert_yaxis()
plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
plt.legend(loc='upper left')
plt.xlabel("Numero di azioni necessarie per coprire lo state space")
plt.ylabel("% di azioni con reward")

plt.show()
