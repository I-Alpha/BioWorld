
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
    'herbivore' :50,
    'carnivore': 25,    
}

settings['carnivore'] = {
    "bite_dmg" : .02,
    "digest_efficiency" : .50,
    "digest_time" : 1,
    "r_fitness_decay" : 0 ,
    "v_fitness_decay" : 0 ,
    "t_fitness_decay" : .9996,
    "feed_range" : 0.065,
}

settings['herbivore'] = {
    "digest_efficiency" : .80,
    "digest_time" : 1,
    "bite_dmg" : .007,
    "t_fitness_decay" : .9996,
    "r_fitness_decay" : 1,
    "v_fitness_decay" : 1, 
    "feed_range" : 0.065,
}

settings['omnivore'] = {}
settings["pre-train_time"]={}
settings["mind_type"]="custom" 
settings["neuralnet"] = None
settings["pre-train_time"]["step"] = 5
settings["pre-train_time"]["mind_type"] = "random"
settings['gen_time'] = 150  # generation length  (seconds)
settings['dt'] = 0.4  
settings['gens'] = 100           # number of generations
settings['elitism'] = 0.20      # elitism (selection bias)
settings['mutate'] = 0.2       # mutation rate

settings=getConfig("default",settings)


def execute():
    print(settings)
    if settings:
        env = get_env("basic")(settings) 
        env.prepare()
        evolve = get_algo("basic")
    
    for gen in range(0, settings['gens']):
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
    else:
        print("No settings detected")

if __name__ == '__main__':
    execute() 
