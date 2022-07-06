
from evolution.get_evolution_algo import get_algo
from environments.get_environment import get_env
from settings.config import getConfig
 
settings={} 
settings['FPS'] = 80
settings['graphics'] = True
settings["world_size"] = (2) 
settings["dash_debug"] = True
settings["starting_interval"] = 0
settings['screen_size'] = (900,800)
settings["dash_update_interval"] = 100
settings["pygame_worldbox"] = [(settings['screen_size'][0] * .01, settings['screen_size'][1] *.1),
                                (settings['screen_size'][0] * .7, settings['screen_size'][1] *.8),]

settings['agents'] = {
    'herbivore' :60,
    'carnivore': 20,    
}

settings['carnivore'] = {
    "bite_dmg" : .002,
    "digest_efficiency" : .05,
    "digest_time" : .01,
    "r_fitness_decay" :.9987,
    "v_fitness_decay" : .997,
    "t_fitness_decay" : .9985,
    "feed_range" : 0.065,
    'food_detect_range': 0.2,
    'neighbour_detect_range': 0.1
}

settings['herbivore'] = {
    "digest_efficiency" : .80,
    "digest_time" : 1,
    "bite_dmg" : .0007,
    "t_fitness_decay" : .9987,
    "r_fitness_decay" : .9999,
    "v_fitness_decay" : .998, 
    "feed_range" : 0.065,
    'food_detect_range': 0.15,
    'neighbour_detect_range': 0.15
}

settings['omnivore'] = {}
settings["pre-train_time"]={}
settings["mind_type"]="custom" 
settings["neuralnet"] = None
settings["pre-train_time"]["step"] = 5
settings["pre-train_time"]["mind_type"] = "random"
settings['gen_time'] = 250  # generation length  (seconds)
settings['dt'] = 0.2  
settings['gens'] = 100           # number of generations
settings['elitism'] = 0.30      # elitism (selection bias)
settings['mutate'] = 0.2       # mutation rate

settings=getConfig("default",settings)


def execute():
    print(settings)
    if settings:
        env = get_env("basic")(settings) 
        env.prepare()
        evolve = get_algo("basic")
    gen = 0 
    while True:#for gen in range(0, settings['gens']):
        # SIMULATE
        env.simulate(settings, gen) 
        
        # EVOLVE organisms
        env.organisms, stats = evolve(settings, env.organisms, gen) 
        if not env.organisms:
            exit("\n\nnot enough orgs left for crossover\n")
        
        env.stats = stats
        
        print('> GEN:', gen, 
              'BEST:{:2f}'.format(stats['BEST']),'AVG:{:2f}'.format(stats['AVG']), 'WORST:{:2f}'.format(stats['WORST']))
        env.populateFood()  
        gen+=1
    else:
        print("No settings detected")

if __name__ == '__main__':
    execute() 
