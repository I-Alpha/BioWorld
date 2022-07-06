import time
from logging import debug
from itertools import groupby
from matplotlib.pyplot import pause
import numpy as np
from math import floor
from math import sqrt
from math import degrees
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import copy
from numpy.core.fromnumeric import shape
from numpy.core.numeric import False_
import pandas as pd
import cv2
from math import atan2
import time
import turtle as tl
import random
import copy
from pandas.core.indexes import interval
from constants import BLACK, GREEN, ORANGE, WHITE
from utils import calc_heading, dist, info_box
from entities.agent import Agent
from entities.food import Food
from environments.plotting import *
from dash.dependencies import Output, Input
from collections import defaultdict
from multiprocessing import Process, Value, Array, Manager
import pygame
import tensorflow as tf
import itertools

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.LogicalDeviceConfiguration(
    memory_limit=None, experimental_priority=None
)
random.seed(7)


class Env():

    def __init__(self, settings):
        self.fig = False
        self.graphics = settings['graphics']
        self.screen_size = settings['screen_size']
        self.start_food = settings['food_num']
        self.start_agents = settings['pop_size']
        self.food_template = settings['food_template']
        self.agent_template = settings['agent_template']
        self.FPS = settings['FPS']
        self.organisms = []
        self.org_dict = {}
        self.gen = 0
        self.foods = []
        self.dead_foods = []
        self.hnodes = settings['hnodes']
        self.onodes = settings['onodes']
        self.inodes = settings['inodes']
        self.settings = settings
        self.time = time.time()
        self.gen_start_time = 0
        self.gen_curr_time = 0
        self.t_step = 0
        self.total_time_steps = 0
        self.g_initiated = False
        self.pygame_background_color = settings["colors"]['background_color']
        self.stats = {
            "BEST": 0,
            "AVG": 0,
            "WORST": 0,
        }
        self.display_vals = self.compute_xy_values()
        if self.graphics:
            self.initialise_display()

    def getXYCoordsFood(self):
        new_data = self.entities
        f_data_x = []
        f_data_y = []
        [[f_data_x.append(item['x']), f_data_x.append(item['y'])]
         for item in new_data if item['type'] == 'food']
        return [f_data_x, f_data_y]

    def getXYCoordsOgranisms(self):
        new_data = self.entities
        o_datax = [item['x']
                   for item in new_data if item['type'] == 'organism']
        o_datay = [item['y']
                   for item in new_data if item['type'] == 'organism']
        return [o_datax, o_datay]

    def update_entities_dict(self):
        global env_app
        entities = []
        # PLOT ORGANISMS
        for organism in self.organisms:
            entities.append(
                {"x": organism.x, "y": organism.y, "type": "organism"})
        #    plot_organism(organism.x, organism.y, organism.r, ax)

        # PLOT FOOD PARTICLES
        for food in self.foods:
            entities.append({"x": food.x, "y":  food.y, "type": "food"})

        self.entities = entities
        env_app = self

    def populateEnv(self):
        self.populateFood()
        self.populateOrganism()

    def populateFood(self):

        # --- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
        self.foods = []
        self.dead_foods = []
        if self.settings["food_mode"] == "default":
            for i in range(0, self.start_food):
                self.foods.append(Food(self.settings))
        else:
            for i in range(0, self.start_food):
                self.foods.append(self.food_template(self.settings))

        self.food_last_spawned_at = time.time()

    def init_bias(self):
        Bh = np.full((1, self.hnodes), 0.1)
        Bo = np.full((1, self.onodes), 0.1)
        return Bh, Bo

    def init_weights(self):
        # mlp weights (input -> hidden)
        wih_init = np.random.uniform(-1, 1,
                                     (self.hnodes, self.inodes))
        # mlp weights (hidden -> output)
        who_init = np.random.uniform(-1, 1,
                                     (self.onodes, self.hnodes,))
        return wih_init, who_init

    def populateOrganism(self):
        # --- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
        self.organisms = []
        self.dead_organisms = 0
        for key in self.settings["agents"]:
            (wih_init, who_init) = self.init_weights()
            for x in range(self.settings["agents"][key]):
                agent = Agent(self.settings, wih_init, who_init,
                              name='gen['+str(x)+']-org['+key+'-'+str(x)+']', behaviour=key,
                              mind_type=self.settings["mind_type"])
                self.organisms.append(agent)

    def get_org_dict(self):
        org_dict = {}
        for key in self.settings["agents"]:
            org_dict[key] = [
                org for org in self.organisms if org.behaviour == key]
        return org_dict

    def prepare(self):
        self.populateEnv()
        self.update_entities_dict()
        if self.graphics and not self.g_initiated:
            self.initialise_display()
            self.g_initiated = True
            time.sleep(0.5)

    def compute_xy_values(self):
        x_diff = self.settings["x_max"]-self.settings["x_min"]
        y_diff = self.settings["y_max"]-self.settings["y_min"]

        x_ratio = (self.settings["pygame_worldbox"][1][0]/x_diff)
        y_ratio = (self.settings["pygame_worldbox"][1][1]/y_diff)

        xUp = 0.1
        yUp = 0.55

        range_modifierx = ((x_diff)/2) + xUp
        range_modifiery = ((y_diff)/2) + yUp

        range_multiplierx = .980 * \
            self.settings['pygame_worldbox'][1][0]/x_diff
        range_multipliery = .980 * \
            self.settings['pygame_worldbox'][1][1]/y_diff
        return {
            "x_ratio": x_ratio,
            "y_ratio": y_ratio,
            "range_modifierx": range_modifierx,
            "range_modifiery": range_modifiery,
            "range_multiplierx": range_multiplierx,
            "range_multipliery": range_multipliery
        }

    def initialise_display(self):
        pygame.font.init()
        self.screen = pygame.display
        self.myfont = pygame.font.SysFont('Comic Sans MS', 20)
        self.surface = self.screen.set_mode(self.screen_size)
        self.screen.set_caption(' BIO - WORLD ')
        self.clock = pygame.time.Clock()

    def update_display(self, t_step):
        def draw_info_box(info_box):
            td = self.settings["pygame_worldbox"][0][1]
            for i, arr in enumerate(info_box):
                textsurface = self.myfont.render(
                    ' '.join(arr), False, (0, 0, 0))
                self.surface.blit(
                    textsurface,
                    textsurface.get_rect(
                        x=(self.settings["pygame_worldbox"][1][0])*1.05,
                        y=td
                    )
                )
                td += ((self.settings["pygame_worldbox"][0][1])/4)

        def draw_organisms():
            for org in self.organisms:
                pygame.draw.circle(
                    self.surface,
                    self.settings["colors"][org.behaviour],
                    (
                        (self.display_vals["range_modifierx"] + org.x) *
                        self.display_vals["range_multiplierx"],
                        (self.display_vals["range_modifiery"] + org.y) *
                        self.display_vals["range_multipliery"]
                    ),
                    floor(
                        org.size * (self.display_vals["x_ratio"] + self.display_vals["y_ratio"]) / 2),
                    floor(
                        org.size * (self.display_vals["x_ratio"] + self.display_vals["y_ratio"]) / 2)
                )

        def draw_background():
            self.surface.fill(WHITE)
            pygame.draw.rect(self.surface, BLACK,
                             pygame.Rect(
                                 self.settings["pygame_worldbox"][0][0],
                                 self.settings["pygame_worldbox"][0][1],
                                 self.settings["pygame_worldbox"][1][0],
                                 self.settings["pygame_worldbox"][1][1]
                             ),
                             1)

        def draw_top_text_line():
            stats = '> GEN: {0:.2f} BEST:{1:.2f} AVG:{2:.2f} WORST:{3:.2f}'.format(
                self.gen, self.stats['BEST'], self.stats['AVG'], self.stats['WORST'])

            textsurface = self.myfont.render(stats, False, (0, 0, 0))

            self.surface.blit(
                textsurface,
                textsurface.get_rect(
                    x=self.settings["pygame_worldbox"][0][0],
                    y=self.settings["pygame_worldbox"][0][1]/4
                )
            )

        def draw_food():
            for food in self.foods:
                pygame.draw.rect(
                    self.surface,
                    self.settings["colors"]["food"],
                    pygame.Rect(
                        (self.display_vals["range_modifierx"] + food.x) *
                        self.display_vals["range_multiplierx"],
                        (self.display_vals["range_modifiery"] + food.y) *
                        self.display_vals["range_multipliery"],
                        floor(
                            food.size * (self.display_vals["x_ratio"] + self.display_vals["y_ratio"])),
                        floor(
                            food.size * (self.display_vals["x_ratio"] + self.display_vals["y_ratio"]))
                    )
                )

        # DRAW FUNCTIONS ARE CALLED HERE!!

        draw_background()
        draw_info_box(info_box(self, t_step))
        draw_top_text_line()
        draw_organisms()
        draw_food()
        self.screen.update()

    def run_step(self, settings, t_step):
        self.t_step = t_step

        def attempt_kill_organism(org):
            if org.kill or org.fitness <= 0.1:
                self.organisms.remove(org)
                self.dead_organisms += 1
                return True
            return False

        def clean_all_organisms():
            for org in self.organisms:
                attempt_kill_organism(org)

        def clean_all_food():
            for food in [food for food in self.foods if food.kill]:
                self.foods.remove(food)
                self.dead_foods.append(food)

        def handle_non_predator_feeding():
            org_dict = self.get_org_dict()
            non_carnivores = org_dict.get(
                "herbivore", []) + self.org_dict.get("omnivore", [])
            for food in self.foods:
                for org in non_carnivores:
                    food_org_dist = dist(org.x, org.y, food.x, food.y)
                    # UPDATE FITNESS FUNCTION
                    if food_org_dist <= settings[org.behaviour]['feed_range']:
                        org.fitness += food.energy * \
                            settings['herbivore']["digest_efficiency"]
                        food.kill = True
                        food.death_time = t_step
                        org.d_food = 100
                        org.r_food = 0

        def handle_predator_feeding():
            # UPDATE PREY BEHAVIOR FUNCTION
            org_dict = self.get_org_dict()
            carnivores = org_dict.get('carnivore', [])
            herbivores = org_dict.get('herbivore', [])
            omnivores = org_dict.get('omnivore', [])

            if len(carnivores) > 0 and len(herbivores) > 0:
                for prey in herbivores + omnivores:
                    for carnivore in carnivores:
                        carnivore.d_food = 100
                        carnivore.r_food = 0
                        herbivore_dist = dist(
                            carnivore.x, carnivore.y, prey.x, prey.y)
                        # DETECT FOOD
                        if herbivore_dist <= self.settings['carnivore']['feed_range']:
                            # UPDATE FITNESS FUNCTION
                            prey.fitness -= settings["carnivore"]["bite_dmg"]
                            carnivore.fitness += prey.fitness * \
                                settings["carnivore"]["digest_efficiency"]
                            if t_step - prey.hlocked_t > settings["hlocked_time"]:
                                prey.fitness -= .3
                                prey.hlocked_t = t_step
                            if prey.fitness <= 0:
                                prey.kill = True
                                prey.death_time = t_step
                            carnivore.digest_t_step = settings["carnivore"]["digest_time"] + t_step

        def respawn_food():
            for food in self.dead_foods:
                time_passed = (t_step - food.death_time)
                if time_passed >= food.respawn_delay:
                    food.respawn(settings)
                    self.dead_foods.remove(food)
                    self.foods.append(food)

        def set_nearest_food(food, org):
            food_org_dist = dist(org.x, org.y, food.x, food.y)
            if food_org_dist < org.d_food:
                org.d_food = food_org_dist
                org.r_food = calc_heading(org, food)

        def compute_perceptions_data():
            org_dict = self.get_org_dict()

            def set_non_pred_nearest_food():
                # COMPUTE FOOD
                carnivores = org_dict.get('carnivore', False)
                herbivores = org_dict.get('herbivore', False)
                omnivores = org_dict.get('omnivore', False)
                omnivores = omnivores if omnivores else []
                for food in self.foods:
                    if not food.kill:
                        for org in herbivores + omnivores:
                            set_nearest_food(food, org)

            def set_nearest_neighbour():
                # CALCULATE HEADING TO NEAREST NEIGHBOR
                for org1, org2 in itertools.permutations(self.organisms, 2):
                    org1_org2_dist = dist(org1.x, org1.y, org2.x, org2.y)
                    if org1_org2_dist <= 0.1:
                        org1.neighbours += 1
                        if (org1_org2_dist < org1.nearest_neighbour_d):
                            org1.nearest_neighbour_d = org1_org2_dist
                            org1.nearest_neighbour_v = org2.v
                            org1.nearest_neighbour_r = calc_heading(org1, org2)
                            org1.nearest_neighbour_type = org2.b_id + 1
                        if (org1.behaviour == "carnivore" and org2.behaviour != "carnivore"):
                            if not org2.kill:
                                set_nearest_food(org2, org1)

            set_non_pred_nearest_food()
            set_nearest_neighbour()

        def organisms_think_act():
            for org in self.organisms:
                org.think()

            # update org position
            for org in self.organisms:
                org.update_r(settings)
                org.update_vel(settings)
                org.update_pos(settings)

        clean_all_food()

        # UPDATE FITNESS FUNCTION
        handle_non_predator_feeding()
        handle_predator_feeding()

        clean_all_food()
        clean_all_organisms()
        respawn_food()

        if len(self.organisms) < 2:
            exit("\n\nNo more organisms left. Ending simulation\n")
        self.resetOrgsStates()
        compute_perceptions_data()
        organisms_think_act()

        self.update_entities_dict()
        self.gen_curr_time = (time.time() - self.gen_start_time)/1000

        if self.graphics:
            self.update_display(t_step)

    def resetOrgsStates(self):
        for org in self.organisms:
            org.d_food = 100
            org.r_food = 0
            org.neighbours = 0
            org.nearest_neighbour_d = 100
            org.nearest_neighbour_v = 0
            org.nearest_neighbour_r = 0
            org.nearest_neighbour_type = 0

    def simulate(self, settings, gen):

        global env_app
        self.gen = gen
        self.update_entities_dict()
        self.dead_organisms = 0
        self.gen_start_time = time.time()
        total_time_steps = int(settings['gen_time'] / settings['dt'])
        self.total_time_steps = total_time_steps
        # --- CYCLE THROUGH EACH TIME STEP ---------------------+
        self.gen_start_time = time.time()
        for t_step in range(0, total_time_steps, 1):
            self.run_step(settings, t_step)
        return self.organisms
