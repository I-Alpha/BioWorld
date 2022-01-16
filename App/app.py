

from Food import *
from Agent import *
from BioworldEnv import *
from evolve import *
from model.NNmodels import *
# --- CONSTANTS ----------------------------------------------------------------+

settings = {}

# GRAPHICS 

settings['graphics'] = True
settings['screen_size'] = (900,900)
settings['FPS'] = 80
settings["dash_update_interval"] = 100
settings["starting_interval"] = 0
settings["world_size"] = (2) 
settings["pygame_worldbox"] = [(settings['screen_size'][0] * .01, settings['screen_size'][1] *.1),
                               (settings['screen_size'][0] * .7, settings['screen_size'][1] *.8),]


settings["dash_debug"] = True
# EVOLUTION SETTINGS
 

               
            


settings['agents'] = {
    'herbivore' : 50,
    'predator': 20,    
}

settings["hlocked_time"] = 100
settings['pop_size'] = np.sum([i for i in settings['agents'].values()])

settings['gens'] = 10000          # number of generations
settings['elitism'] = 0.20      # elitism (selection bias)
settings['mutate'] = 0.2       # mutation rate

# SIMULATION SETTINGS
settings['gen_time'] = 100 # generation length         (seconds)
settings['dt'] = 0.04           # simulation time step      (dt)

settings['food_respawn_time']= 1000 
settings["food_num"] = 50     # number of food particles
settings["food_decay per_step"] = 0#.000005
settings["nn_dr_decay"] =  0#.000015#settings["food_decay per_step"]*.000005
settings["nn_dv_decay"] =  0#.000015#settings["food_decay per_step"]*.00001
settings["resistance"] = .999 
settings['wall_penalty'] = 0.005

# max rotational speed      (degrees per second)
settings['dr_max'] = 720 
settings['v_max'] = 0.5        # max velocity              (units per second)
settings['dv_max'] = 0.35      # max acceleration (+/-)    (units per second^2)

settings['x_min'] = -2.0        # arena western border
settings['x_max'] = 2.0        # arena eastern border
settings['y_min'] = -2.0        # arena southern border
settings['y_max'] = 2.0        # arena northern border

settings["food_template"] = Food
settings["agent_template"] = Agent
  
settings['plot'] = False        # plot final generation?

# ORGANISM NEURAL NET SETTINGS

settings['gravity'] = 1.9  # default
  
settings['neuralnet'],_ = build_model1()
settings['org_mode'] = settings['mind_type'] = "default" 

settings["mass"] = 2
settings["acc_min"] = .2
settings["acc_max"] = 3

settings["food_mode"] = "default" 
 
settings["action_space"] = 2
settings["state_space"] = len((settings["agent_template"])(settings).getState()) 
settings['inodes'] = settings["state_space"]        # number of input nodes
settings['hnodes'] = 6       # number of hidden nodes
settings['onodes'] = 2          # number of output nodes 
# state_sapce
# agent x.y
# y
## colours
settings["organisms_color"] = "#121212" 
settings["food_color"] = "#4f7507" 
settings['pygame_background_color'] = 'white'

def run(settings):
    env = JumpEnv(settings) 
    env.prepare()
    for gen in range(0, settings['gens']):
        # SIMULATE
        env.organisms = env.simulate(settings, gen) 
        # EVOLVE organisms
        
        env.organisms, stats = evolve(settings, env.organisms, gen) 
        if not env.organisms:
            exit("\n\nnot enough orgs left for crossover\n")
        
        env.stats = stats
        print('> GEN:', gen, 
              'BEST:{:2f}'.format(stats['BEST']),'AVG:{:2f}'.format(stats['AVG']), 'WORST:{:2f}'.format(stats['WORST']))
        env.populateFood() 
    pass
 

if __name__ == '__main__':
    run(settings) 
