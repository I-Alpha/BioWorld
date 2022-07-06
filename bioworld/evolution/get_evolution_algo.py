from evolution import basic_ev, default_ev

def get_algo(ev_name:str): 
    evolution:function = {
        "default": default_ev.evolve,
        "basic": basic_ev.evolve,
    }
    return evolution.get(ev_name,False)
