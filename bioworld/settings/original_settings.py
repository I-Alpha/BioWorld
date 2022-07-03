import numpy as np
from entities.agent import Agent
from entities.food import Food
from models.nn_models import build_simple_model
from constants import BLACK, GREEN, ORANGE, RED, WHITE

def getSettings(settings):
    _settings={}
    _settings.update(settings)
    _settings["colors"] = {} if not settings.get(
        "colors", False) else settings["colors"]
    _settings["colors"]["herbivore"] = BLACK
    _settings["colors"]["carnivore"] = ORANGE
    _settings["colors"]["food"] = GREEN
    _settings["colors"]['background_color'] = WHITE

    # EVOLUTION SETTINGS 
    _settings["hlocked_time"] = 100
    _settings['pop_size'] = np.sum([i for i in _settings['agents'].values()]) 
    _settings['gens'] = 10000          # number of generations
    _settings['elitism'] = 0.20      # elitism (selection bias)
    _settings['mutate'] = 0.2       # mutation rate

    # SIMULATION SETTINGS
    _settings['gen_time'] = 100 # generation length         (seconds)
    _settings['dt'] = 0.04           # simulation time step      (dt)

    _settings['food_respawn_time']= 1000 
    _settings["food_num"] = 50     # number of food particles
    _settings["food_decay per_step"] = 0#.000005
    _settings["nn_dr_decay"] =  0#.000015#settings["food_decay per_step"]*.000005
    _settings["nn_dv_decay"] =  0#.000015#settings["food_decay per_step"]*.00001
    _settings["resistance"] = .999 
    _settings['wall_penalty'] = 0.005

    # max rotational speed      (degrees per second)
    _settings['dr_max'] = 720 
    _settings['v_max'] = 0.5        # max velocity              (units per second)
    _settings['dv_max'] = 0.35      # max acceleration (+/-)    (units per second^2)

    _settings['x_min'] = -2.0        # arena western border
    _settings['x_max'] = 2.0        # arena eastern border
    _settings['y_min'] = -2.0        # arena southern border
    _settings['y_max'] = 2.0        # arena northern border

    _settings["food_template"] = Food
    _settings["agent_template"] = Agent
    
    _settings['plot'] = False        # plot final generation?

    # ORGANISM NEURAL NET SETTINGS

    _settings['gravity'] = 1.9  # default
    
    _settings['neuralnet'],_ = build_simple_model()
    _settings['org_mode'] = _settings['mind_type'] = "default" 

    _settings["mass"] = 2
    _settings["acc_min"] = .2
    _settings["acc_max"] = 3

    _settings["food_mode"] = "default" 
    
    _settings["action_space"] = 2
    _settings["state_space"] = len((_settings["agent_template"])(_settings).getState()) 
    _settings['inodes'] = _settings["state_space"]        # number of input nodes
    _settings['hnodes'] = 6       # number of hidden nodes
    _settings['onodes'] = 2   # number of output nodes

    _settings["organisms_color"] = "#121212" 
    _settings["food_color"] = "#4f7507" 
    _settings['pygame_background_color'] = 'white'
    _settings.update(settings)

    return _settings 