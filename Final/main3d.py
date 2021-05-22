import random
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib
from path import Path
import pandas as pd
import glob2

from matplotlib.figure import Figure
import sys
import errno
from tkinter import *
# matplotlib.use("TkAgg")
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from PIL import Image, ImageTk

import tkinter as tk                # python 3
from tkinter import font as tkfont  # python 3

import matplotlib.pyplot as plt

import logging
from enum import Enum, auto

from deliveryMap3D import DeliveryMap, Render
from qTable import QTableModel
from qTableTrace3D import QTableTraceModel

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level


# Enumeración que contiene todos las formas posibles de resolver el problema y asigna a cada uno un número entero empezando por el 1.
class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    RANDOM_MODEL = auto()
    Q_ELIGIBILITY = auto()
    

matrix = np.array([[0.7, 0.6, 1.,  0.,  1.,  0.8],
                   [1.,  0.3, 0.,  0.2, 0.,  0. ],
                   [0.,  0.,  0.,  0.,  0.,  0. ],
                   [0.,  0.5, 0.2, 0.,  0.6, 0.4],
                   [0.2, 0.4, 1.,  0.3, 0.,  1. ],
                   [0.8, 0.,  1.,  0.,  1.,  0 ]])

heightsMatrix = np.array([[1,   3,   2,   0,   2,   3],
                          [1,   3,   0,   4,   0,   0],
                          [0,   0,   0,   0,   0,   0],
                          [0,   1,   2,   0,   1,   2],
                          [4,   2,   4,   3,   0,   1],
                          [4,   0,   4,   0,   3,   0]])

storageCell = (1, 1)
deliveryCell = (5, 5)

'''Parámetros de estudio'''
parametros = {}
'''discount: (gamma) preference for future rewards (0 = not at all, 1 = only)'''
parametros['discount'] = 0.90 #Predeterminado: 0.9
'''exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)'''
parametros['exploration_rate'] = 0.10 #Predeterminado: 0.10
'''exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)'''
parametros['exploration_decay'] = 0.995 #Predeterminado: 0.995
'''learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)'''
parametros['learning_rate'] = 0.10 #Predeterminado: 0.10
'''eligibility_decay: (lambda) eligibility trace decay rate per step (0 = no trace, 1 = no decay)'''
parametros['eligibility_decay'] = 0.80 #Predeterminado: 0.80
    

test = Test.Q_ELIGIBILITY

#Mapa en forma de np.array
arrayMap = matrix
heightsMap = heightsMatrix

#Se crea el juego creando un objeto de la clase DeliveryMap
deliverySystem = DeliveryMap(arrayMap, heightsMap, storageCell, 0)

# only show the maze
if test == Test.SHOW_MAZE_ONLY:
    deliverySystem.render(Render.MOVES)
    deliverySystem.reset()

# Entrenamiento del agente utilizando el método Q-learning
# if test == Test.Q_LEARNING:
#     deliverySystem.render(Render.TRAINING)
#     model = QTableModel(deliverySystem, name="QTableModel")
#     h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=250,
#                               stop_at_convergence=True)

# train using tabular Q-learning and an eligibility trace (aka TD-lamba)
if test == Test.Q_ELIGIBILITY:
    deliverySystem.render(Render.TRAINING)
    model = QTableTraceModel(deliverySystem)
    h, w, episodes, _ = model.train(discount=parametros['discount'], exploration_rate=parametros['exploration_rate'],
                                    learning_rate = parametros['learning_rate'], exploration_decay = parametros['exploration_decay'],
                                    eligibility_decay = parametros['eligibility_decay'], stop_at_convergence=True)

# draw graphs showing development of win rate and cumulative rewards
try:
    h  # force a NameError exception if h does not exist, and thus don't try to show win rate and cumulative reward
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.canvas.set_window_title(model.name)
    ax1.plot(*zip(*w))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    ax2.plot(h)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except NameError:
    pass

deliverySystem.render(Render.MOVES)
# deliverySystem.play(model, start_cell=(0, 0))
# deliverySystem.play(model, start_cell=(2, 5))
status, totalReward, no_steps = deliverySystem.play(model, deliveryCell[::-1], print_route = True)

plt.show()  # must be placed here else the image disappears immediately at the end of the program