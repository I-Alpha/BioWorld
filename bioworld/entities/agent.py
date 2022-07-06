from random import uniform
import numpy as np
import operator
from icecream import *
from random import sample
from math import cos, floor, sqrt
from math import radians
from math import sin
from random import uniform
from collections import defaultdict
import pygame
from keras.backend import clear_session
from time import time
from math import atan2
from math import degrees
from types import SimpleNamespace
from models.nn_models import build_simple_model


def timeit(func, arg, iterations):
    t0 = time()
    for _ in range(iterations):
        func(arg)
    print("%.4f sec" % (time() - t0))


bids = ["carnivore", "herbivore", "omnivore"]


class Agent():
    def __init__(self, settings, wih=None, who=None, name=None, neuralnet=None, actionspace=None, behaviour="herbivore", mind_type="default",w_new=False, position=(False, False)):
        self.x = position[0] if position[0] else uniform(
            settings['x_min'], settings['x_max'])  # position (x)
        self.y = position[1] if position[1] else uniform(
            settings['y_min'], settings['y_max'])  # position (y)
        self.r = uniform(0, 360)                 # orientation   [0, 360]
        self.v = uniform(0, settings['v_max'])   # velocity      [0, v_max]
        self.dv = uniform(-settings['dv_max'], settings['dv_max'])   # dv
        self.d_food = 100   # distance to nearest food
        self.r_food = 0     # orientation to nearest food
        self.fitness = 1    # fitness (food count)
        self.neighbours = 0
        self.nearest_neighbour_r = 0
        self.nearest_neighbour_v = 0
        self.nearest_neighbour_type = 0
        self.nearest_neighbour_d = 100
        self.wih = wih
        self.who = who
        self.type = "agent"
        self.kill = False
        self.name = name
        self.t_fitness_decay = settings[behaviour]['t_fitness_decay'] or 1
        self.t_step = 0
        self.hlocked_t = 0
        self.digest_t_step = 0 
        self.mind_type = mind_type
        self.behaviour = behaviour
        self.b_id = bids.index(behaviour)
        self.actionspace = settings["action_space"]
        self.statespace = len(self.getState())
        if mind_type == "custom":
            if neuralnet == None or neuralnet == "":
                self.brain, self.nn_weights = build_simple_model(
                    self.actionspace, self.statespace)
                if  w_new is not False:
                    self.nn_weights = w_new
                    self.brain.set_weights(w_new)
            else:
                self.brain = neuralnet
                self.nn_weights = neuralnet.get_weights()
        self.size = (.02)

    # NEURAL NETWORK FUNCTIONS
    def think(self):
        self.fitness *= self.t_fitness_decay
        state = self.getState()
        if self.mind_type == "random":
            out = floor(uniform(0, self.actionspace))
            # [-1, 1]  (accelerate=1, deaccelerate=-1)
            self.nn_dv = float(uniform(-1, 1))
            self.nn_dr = float(uniform(-1, 1))  # (left=1, right=-1)
            return
        if self.mind_type == "custom":
            input = np.array(state)
            input = input.reshape(1,self.statespace)
            out = self.brain(input).numpy()[0]
            # UPDATE dv AND dr WITH MLP RESPONSE
            # [-1, 1]  (accelerate=1, deaccelerate=-1)
            self.nn_dv = float(out[0])
            # [-1, 1] (accelerate=1)   # [-1, 1]  (left=1, right=-1)
            self.nn_dr = float(out[1])
            return
        if self.mind_type == "default":
            # SIMPLE MLP
            def af(x): return np.tanh(x)              # activation function
            # h1 = af( self.wih @ state)  # hidden layer
            h1 = af(np.dot(self.wih, state))  # hidden layer
            out = af(np.dot(self.who, h1))          # output layer

            # UPDATE dv AND dr WITH MLP RESPONSE
            # [-1, 1]  (accelerate=1, deaccelerate=-1)
            self.nn_dv = float(out[0])
            self.nn_dr = float(out[1])   # [-1, 1]  (left=1, right=-1)
            return

    def getState(self):
        return [
            self.d_food *.01,
            self.r_food,
            self.v,
            self.fitness*.01,
            self.nearest_neighbour_r,
            self.nearest_neighbour_v,
            self.nearest_neighbour_d *.01,
            self.nearest_neighbour_type/3,
            (1/self.neighbours) if self.neighbours else 0,
        ]

    # UPDATE HEADING
    def update_r(self, settings):
        if self.nn_dr > 0:
            self.fitness = self.fitness if not settings[self.behaviour][
                "r_fitness_decay"] else (self.fitness*settings[self.behaviour]["r_fitness_decay"])

        self.r += self.nn_dr * settings['dr_max'] * settings['dt']
        self.r = self.r % 360

    # UPDATE VELOCITY
    def update_vel(self, settings):
        if self.nn_dv > 0:
            self.fitness = self.fitness if not settings[self.behaviour]["v_fitness_decay"] else (
                self.fitness*settings[self.behaviour]["v_fitness_decay"])
        self.v += self.nn_dv * settings['dv_max'] * settings['dt']
        if self.t_step < self.digest_t_step:
            self.v *= .6
        self.v *= settings["resistance"]
        if self.v < 0:
            self.v = 0
        if self.v > settings['v_max']:
            self.v = settings['v_max']

    def update_pos(self, settings):
        dx = self.v * cos(radians(self.r)) * settings['dt']
        dy = self.v * sin(radians(self.r)) * settings['dt']

        self.x += dx
        self.y += dy

        if self.x < settings['x_min']:
            self.x = settings['x_min']
            self.fitness -= settings['wall_penalty']

        if self.x > settings['x_max']:
            self.x = settings['x_max']
            self.fitness -= settings['wall_penalty']

        if self.y < settings['y_min']:
            self.y = settings['y_min']
            self.fitness -= settings['wall_penalty']

        if self.y > settings['y_max']:
            self.y = settings['y_max']
            self.fitness -= settings['wall_penalty']
