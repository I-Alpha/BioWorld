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
import sys
from pandas.core.indexes import interval
from plotting import *
import dash
from dash.dependencies import Output, Input
import plotly
from collections import defaultdict
from Agent import Agent
from Food import Food
import multiprocessing 
from multiprocessing import Process, Value, Array, Manager
import webbrowser
import pygame
import tensorflow as tf
import itertools

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.LogicalDeviceConfiguration(
    memory_limit=None, experimental_priority=None
) 

# pygame.font.init()
# pygame.init()

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  return "{0}:{1}:{2:0.2f}".format(int(hours),int(mins),sec)
 
def infoBox(env_app,n_intervals): 
    return [
        'time: {}'.format(time_convert(time.time() - env_app.time)),
        'gen_time: {}'.format(time_convert(env_app.gen_curr_time)),
        'live_organisms: {0:.2f}'.format(len(env_app.organisms)),
        'dead_organisms: {0:.2f}'.format(env_app.dead_organisms),
        'time_step: {0:0.2f}/{1}'.format(env_app.t_step,env_app.total_time_steps),
        'total: {0:0.2f}'.format(len(env_app.foods)+len(env_app.organisms)),
        'gen: {0:0.2f}'.format( env_app.gen),
        'food_len: {0:0.2f}'.format(len(env_app.foods)),
        'dash_intervals: {0:.2f}'.format(n_intervals),
    ]
            
def dist(x1, y1, x2, y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)
 
def calc_heading(org, food):
    d_x = food.x - org.x
    d_y = food.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180:
        theta_d += 360
    return theta_d / 180


class JumpEnv():

    random.seed(6)
    # screen = pygame.display.set_mode((300,300))
    ORANGE = pygame.Color(255,165,0)
    GREEN = pygame.Color(0,255,0)
    WHITE = pygame.Color(255,255,255)
    BLACK = pygame.Color(0,0,0)
    def __init__(self, settings):
        self.fig = False
        self.graphics = settings['graphics']
        self.screen_size = settings['screen_size']
        self.start_food = settings['food_num']
        # self.start_obstacles = settings['start_obstacles']
        self.start_agents = settings['pop_size']
        self.food_template = settings['food_template']
        self.agent_template = settings['agent_template']
        self.FPS = settings['FPS']
        self.organisms = [] 
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
        self.t_step =0 
        self.total_time_steps=0   
        self.g_initiated=False       
        self.pygame_background_color = settings['pygame_background_color']
        self.stats = {
              "BEST":0,
              "AVG":0,
              "WORST":0,
        }
        if self.graphics:
            self.initialise_display()
                
    def prepare(self):
        self.populateEnv()
        self.update_entities()
        if self.graphics and not self.g_initiated: 
            self.initialise_display()        
            self.g_initiated=True
            time.sleep(0.5)      
           
    def initialise_display(self):  
        pygame.font.init()
        self.screen = pygame.display
        self.myfont = pygame.font.SysFont('Comic Sans MS', 20)
        self.surface = self.screen.set_mode(self.screen_size) 
        self.screen.set_caption(' BIO - WORLD ')  
        self.clock = pygame.time.Clock()
    
    def update_display(self,t_step):  
        self.surface.fill(JumpEnv.WHITE) 
        pygame.draw.rect(self.surface, JumpEnv.BLACK, 
                         pygame.Rect(
                             self.settings["pygame_worldbox"][0][0],
                             self.settings["pygame_worldbox"][0][1],
                             self.settings["pygame_worldbox"][1][0], 
                             self.settings["pygame_worldbox"][1][1] 
                             ),
                         2
                         )
        infobox = infoBox(self,t_step) 
        
        td  = self.settings["pygame_worldbox"][0][1]
        for i,arr in enumerate(infobox):
            textsurface = self.myfont.render(' '.join(arr) ,False, (0, 0, 0))
            self.surface.blit(
                textsurface,
                textsurface.get_rect(
                    x =(self.settings["pygame_worldbox"][1][0])*1.05,
                    y=td 
                )
            )
            td+=((self.settings["pygame_worldbox"][0][1])/4)
         
        stats='> GEN: {0:.2f} BEST:{1:.2f} AVG:{2:.2f} WORST:{3:.2f}'.format(self.gen,self.stats['BEST'],self.stats['AVG'],self.stats['WORST'])
        textsurface = self.myfont.render( stats ,False, (0, 0, 0))
        self.surface.blit(
            textsurface,
            textsurface.get_rect(
                x = self.settings["pygame_worldbox"][0][0],
                y = self.settings["pygame_worldbox"][0][1]/4
            )
        )
        
        arena_diffx = self.settings["x_max"]-self.settings["x_min"]
        arena_diffy = self.settings["y_max"]-self.settings["y_min"]
        
        x_coeff=(self.settings["pygame_worldbox"][1][0]/arena_diffx)
        y_coeff=(self.settings["pygame_worldbox"][1][1]/arena_diffy)
        
        xUp= 0.00015873015873015873*self.settings["pygame_worldbox"][1][0]
        yUp= 0.0007638888888888889*self.settings["pygame_worldbox"][1][1]
        
        range_modifierx = (arena_diffx)/2 + xUp
        range_modifiery = (arena_diffy)/2 + yUp
        
        range_multiplierx =  .980 * self.settings['pygame_worldbox'][1][0]/arena_diffx
        range_multipliery =  .980 * self.settings['pygame_worldbox'][1][1]/arena_diffy
        
        for org in self.organisms:  
            if org.behaviour=="predator":
                orgColor=JumpEnv.ORANGE
            else:
                orgColor=JumpEnv.BLACK
            pygame.draw.circle(
                self.surface,
                orgColor,
                (   
                   (range_modifierx + org.x) * range_multiplierx ,
                   (range_modifiery + org.y) * range_multipliery
                ),
                floor(org.size * (x_coeff + y_coeff) / 2), 
                floor(org.size * (x_coeff + y_coeff) / 2)
            )
        for food in self.foods:                
            pygame.draw.rect(
                self.surface,
                JumpEnv.GREEN,
                pygame.Rect( 
                   (range_modifierx + food.x) * range_multiplierx,
                   (range_modifiery + food.y) * range_multipliery,
                   floor(food.size * (x_coeff + y_coeff)),
                   floor(food.size * (x_coeff + y_coeff))
                )
            )
            
        self.screen.update()
        
    def getXYCoordsFood(self):
        new_data = self.entities
        f_datax = [item['x'] for item in new_data if item['type'] == 'food']
        f_datay = [item['y'] for item in new_data if item['type'] == 'food'] 
        return [f_datax, f_datay]

    def getXYCoordsOgranisms(self):
        new_data = self.entities
        o_datax = [item['x']
                   for item in new_data if item['type'] == 'organism']
        o_datay = [item['y']
                   for item in new_data if item['type'] == 'organism'] 
        return [o_datax, o_datay]  

    def update_entities(self):
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
        
        self.entities=entities
        env_app=self

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
                
        self.food_last_spawned_at=time.time()

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
        if self.settings["org_mode"] == "default":
            for key in self.settings["agents"]:
                (wih_init, who_init) = self.init_weights()
                for x in range(self.settings["agents"][key]):
                    self.organisms.append(Agent(self.settings, wih_init, who_init,
                                            name='gen['+str(x)+']-org['+key+'-'+str(x)+']', behaviour=key))

        else:
            for key in self.settings["agents"].keys():
                for i in range(self.settings["agents"][key]):
                    self.organisms.append(self.agent_template(
                        self.settings, name='gen['+i+']-org['+key+'-'+str(i)+']nn', neuralnet=self.settings["neuralnet"], actionspace=2, mind_type=self.settings['mind_type'] , behaviour=key))
 
    
    def simulate(self, settings, gen):
        
        global env_app
        self.gen=gen 
        self.update_entities() 
        self.dead_organisms = 0 
        self.gen_start_time = time.time()
        total_time_steps = int(settings['gen_time'] / settings['dt'])
        self.total_time_steps = total_time_steps
        # --- CYCLE THROUGH EACH TIME STEP ---------------------+
        self.gen_start_time = time.time()           
        for t_step in range(0, total_time_steps, 1): 
            self.t_step = t_step
            env_app = self 
            # UPDATE FITNESS FUNCTION
            # Non-living food organism behaviour
            for food in self.foods:
                if not food.kill:
                    non_predators = [org for org in self.organisms if org.behaviour != "predator"]
                    for org in non_predators:        
                            food_org_dist = dist(org.x, org.y, food.x, food.y)
                            # UPDATE FITNESS FUNCTION
                            if food_org_dist <= 0.075:
                                org.fitness += food.energy 
                                # food.respawn(settings)
                                food.kill = True  
                                food.death_time = t_step
                                self.dead_foods.append(food) 
                            
                            # RESET DISTANCE AND HEADING TO NEAREST FOOD SOURCE
                            org.d_food = 100
                            org.r_food = 0
                   
            
            # UPDATE PREY_BEHAVIOUR FUNCTION
            organism_group = {}
            for key, group in groupby(self.organisms, lambda x: x.behaviour=="herbivore"):  
                organism_group[key] = [agent for agent in group]           
             
            if len(organism_group)  >= 2:
                for predator in organism_group[False]:
                    predator.d_food = 100
                    predator.r_food = 0 
                    for herbivore in organism_group[True]:
                        if not herbivore.kill:
                            dist_p_h = dist(predator.x, predator.y, herbivore.x, herbivore.y) 
                            if dist_p_h < 0.075:
                                predator.fitness += (herbivore.fitness * .15) 
                                if t_step - herbivore.hlocked_t > settings["hlocked_time"]:
                                    herbivore.fitness -= .3
                                    herbivore.hlocked_t = t_step 
                                if herbivore.fitness <= 0:
                                    herbivore.kill = True
                                    herbivore.death_time = t_step 
                                predator.digest_t_step = 20 + t_step
                                # food.respawn(settings) 
                                
            #Remove dead food
            for food in self.foods:
                if food.kill: 
                    self.foods.remove(food)
                    
            for org in self.organisms:
                
                #reset neighbouring orgs info
                org.neighbours=0
                org.nearest_neighbour_d = 0
                org.nearest_neighbour_v = 0
                org.nearest_neighbour_r = 0
                org.nearest_neighbour_type = 0 
                
                org.t_step = t_step
                org.size =  .02 + (org.fitness * .0001)  
                
                #Remove dead organisms
                if not org.kill:
                    org.fitness = org.fitness - (self.settings["food_decay per_step"] if org.behaviour != "predator" else (self.settings["food_decay per_step"]/1.3))
                    if org.fitness <= 0:
                        org.kill = True
                if org.kill: 
                    self.organisms.remove(org)  
                    self.dead_organisms += 1
                
            
            if len(self.organisms) < 2:
                exit("\n\nNo more organisms left. Ending simulation\n") 
                
            #Spawn queued food
            for food in self.dead_foods:
                time_passed = (t_step - food.death_time) 
                if time_passed  > food.respawn_delay: 
                    food.respawn(settings)
                    self.foods.append(food)            
            
            for food in self.dead_foods:
                if food.kill == False:  
                    self.dead_foods.remove(food)
                    
            self.update_entities()
            
                 

            #split organism group between predators and non-predators
            non_predators= list(filter(lambda x: x.behaviour != 'predator', self.organisms))
                
            # CALCULATE HEADING TO NEAREST FOOD SOURCE
            for food in self.foods:
                for org in non_predators:

                    # CALCULATE DISTANCE TO SELECTED FOOD PARTICLE
                    food_org_dist = dist(org.x, org.y, food.x, food.y)

                    # DETERMINE IF THIS IS THE CLOSEST FOOD PARTICLE
                    if food_org_dist < org.d_food:
                        org.d_food = food_org_dist
                        org.r_food = calc_heading(org, food)

            # CALCULATE HEADING TO NEAREST neghbour
            for org1, org2 in itertools.permutations(self.organisms, 2):
                org1_org2_dist = dist(org1.x, org1.y, org2.x, org2.y) 
                if org1_org2_dist <= 0.1:
                    org1.neighbours+=1  
                    if (org1_org2_dist < org1.nearest_neighbour_d):
                        org1.nearest_neighbour_d = org1_org2_dist
                        org1.nearest_neighbour_v = org2.v
                        org1.nearest_neighbour_r = calc_heading(org1, org2)
                        org1.nearest_neighbour_type = org2.b_id + 1
                        if (org1.behaviour == "predator" and org2.behaviour != "predator" and org1_org2_dist < org1.d_food):                    
                            org1.d_food = org1.nearest_neighbour_d
                            org1.r_food = org1.nearest_neighbour_r 
                    
            # GET ORGANISM RESPONSE
            for org in self.organisms: 
                org.think()

            # UPDATE ORGANISMS POSITION AND VELOCITY
            for org in self.organisms:
                org.update_r(settings)
                org.update_vel(settings)
                org.update_pos(settings)
                
            self.update_entities()
            
            self.gen_curr_time = (time.time() - self.gen_start_time)/1000
            env_app=self
            if self.graphics:
                self.update_display(t_step)  
        return self.organisms
