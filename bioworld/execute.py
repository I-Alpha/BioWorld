
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
    'carnivore': 0,    
}

settings['carnivore'] = {
    "bite_dmg" : .3,
    "digest_efficiency" : .5,
    "digest_time" : 20,
    "r_fitness_decay" : 0 ,
    "v_fitness_decay" : 0 
}

settings['herbivore'] = {
    "digest_efficiency" : .7,
    "digest_time" : 4,
    "bite_dmg" : 1,
    "t_fitness_decay" : 1,
    "r_fitness_decay" : 1,
    "v_fitness_decay" : 1 
}

settings['omnivore'] = {}

settings['gen_time'] = 10  # generation length  (seconds)
settings['dt'] = 0.004  

settings=getConfig("original",settings)


def execute():
    print(settings)
    if settings:
        env = get_env("basic")(settings) 
        env.prepare()
        evolve = get_algo("basic")
    
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
    else:
        print("No settings detected")

if __name__ == '__main__':
    execute() 
