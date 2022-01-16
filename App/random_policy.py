# this is an example code on using the environment. 
import numpy as np 
from App.BioworldEnv import * 
# import the env class

# create an object of env class

settings={}

def random_policy(): 
    
    settings["graphics"] = True
    settings["screen_size"] = (400,400)
    settings["starting_food"] = 20    
    settings["starting_agents"] = 10
    settings["FPS"] = 200 
    settings["action_space"] = 1
    settings["state_space"] = 4 
    settings["gen_length"] = 10 # seconds of each generation 
    
    env = JumpEnv(settings)   

    for e in range(settings["gen_length"]):
        state = env.reset()
        score = 0

        for i in range(max_steps):
            action = np.random.randint(action_space)
            reward, next_state, done = env.step(action)
            score += reward
            state = next_state
             

if __name__ == '__main__':
    random_policy(1000)
