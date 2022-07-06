from random import uniform


class Food():
    def __init__(self, settings, Position=None):
        self.x = uniform(settings['x_min'], settings['x_max'])   # position (x)
        self.y = uniform(settings['y_min'], settings['y_max'])  # position (y)
        self.energy = 1
        self.size = 0.02
        self.type = "food"
        self.kill = False
        self.respawn_loc_id = 0
        self.death_time = 999999
        self.respawn_delay = uniform(
            settings["food_respawn_time"]/4, settings["food_respawn_time"])

    def respawn(self, settings):
        respawn_loc_x = uniform(settings['x_min'], settings['x_max'])
        respawn_loc_y = uniform(settings['y_min'], settings['y_max'])

        quadrantDict = [
            {"x": (settings['x_min'], 0),
             "y": (settings['y_min'], 0)
             },
            {"x": (0, settings['x_max']),
             "y": (settings['y_min'], 0)
             },
            {"x": (settings['x_min'], 0),
             "y": (0,settings['y_max'])
             },
            {"x": (0,settings['x_max']),
             "y": (0,settings['y_max'])
             }
        ]
        
            
        if self.respawn_loc_id != -1:
           x=quadrantDict[self.respawn_loc_id]['x']
           y=quadrantDict[self.respawn_loc_id]['y']
           respawn_loc_x = uniform(x[0],x[1])
           respawn_loc_y = uniform(y[0],y[1])
            
            
        self.x = respawn_loc_x
        self.y = respawn_loc_y
        self.kill = False
        self.energy = 1
        self.death_time = 999999
        self.respawn_delay = uniform(
            settings["food_respawn_time"]/4, settings["food_respawn_time"])
        self.respawn_loc_id+=1
        if self.respawn_loc_id>3:
            self.respawn_loc_id = 0
        
    def update_marker(self):
        self.marker.x = self.x
        self.marker.y = self.y
