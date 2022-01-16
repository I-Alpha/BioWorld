from random import uniform

class Food():
    def __init__(self, settings, Position=None):
        self.x =  uniform(settings['x_min'], settings['x_max'])   # position (x)
        self.y = uniform(settings['y_min'], settings['y_max'])  # position (y)
        self.energy = 1
        self.size = 0.02
        self.type = "food"
        self.kill = False
        self.death_time = 999999 
        self.respawn_delay = uniform(settings["food_respawn_time"]/4,settings["food_respawn_time"])
    def respawn(self,settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.kill = False
        self.energy = 1
        self.death_time = 999999
        self.respawn_delay = uniform(settings["food_respawn_time"]/4,settings["food_respawn_time"])
    def update_marker(self):
        self.marker.x = self.x
        self.marker.y = self.y