from random import uniform

class Obstacle():
    def __init__(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.speed = 15 # px/second 
        self.acceleration = uniform(settings["acc_min"],settings["acc_max"]) 


    def respawn(self,settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.speed = 15
        self.acceleration = uniform(settings["acc_min"],settings["acc_max"]) 
