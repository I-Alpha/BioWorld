from re import L
import numpy as np
from constants import BLACK, GREEN, ORANGE, RED, WHITE
from entities.agent import Agent
from entities.food import Food
from models.nn_models import build_simple_model


def getSettings(settings):
    # GLOBAL SETTINGS
    
    _settings={} 
    _settings.update(settings)
    
    _settings['FPS'] = 80
    _settings['graphics'] = True
    _settings["world_size"] = (2) 
    _settings["dash_debug"] = True
    _settings["starting_interval"] = 0
    _settings['screen_size'] = (900,900)
    _settings["dash_update_interval"] = 100
    _settings["pygame_worldbox"] = [(settings['screen_size'][0] * .01, settings['screen_size'][1] *.1),
                                  (settings['screen_size'][0] * .7, settings['screen_size'][1] *.8),]
  # COLOR
    _settings["colors"] = {} if not settings.get(
        "colors", False) else settings["colors"]
    _settings["colors"]["herbivore"] = BLACK
    _settings["colors"]["carnivore"] = ORANGE
    _settings["colors"]["food"] = GREEN
    _settings["colors"]['background_color'] = WHITE

  # DIMENSIONS
    _settings['x_min'] = -2.0        # arena western border
    _settings['x_max'] = 2.0        # arena eastern border
    _settings['y_min'] = -2.0        # arena southern border
    _settings['y_max'] = 2.0        # arena northern border

  # PHYSICS
    _settings['dt'] = 0.004           # simulation time step      (dt)
    _settings['gravity'] = 1.9  # default
    _settings['dr_max'] = 720
    _settings['v_max'] = 0.1
    _settings['dv_max'] = 0.1
    _settings["resistance"] = .9999
    _settings['wall_penalty'] = 0.005
    _settings["acc_min"] = .05
    _settings["acc_max"] = .2

  # ENTITIES
    # AGENTS
    _settings["agent_template"] = Agent

    _settings['pop_size'] = np.sum([i for i in settings['agents'].values()])

    # FOOD
    _settings["food_mode"] = "default"
    _settings["food_template"] = Food
    _settings['food_respawn_time'] = 1000
    _settings["food_num"] = 50     # number of food particles
    _settings["food_decay per_step"] = 0  # .000005
    _settings["hlocked_time"] = 2

  # NEURALNET SETTINGS
    _settings['org_mode'] = _settings['mind_type'] = _settings.get('mind_type',"default")
    
    _settings["mass"] = 2

    _settings["action_space"] = 2
    _settings["state_space"] =  (_settings["agent_template"])(_settings).statespace 
    # number of input nodes
    _settings['inodes'] = _settings["state_space"]
    _settings['hnodes'] = 14     # number of hidden nodes
    _settings['onodes'] = 2   # number of output nodes

    # EVOLUTION SETTINGS
    _settings['gens'] = 10000          # number of generations
    _settings['elitism'] = 0.20      # elitism (selection bias)
    _settings['mutate'] = 0.2       # mutation rate

  # PLOT
    _settings['plot'] = False        # plot final generation?

    _settings.update(settings)
    return _settings
