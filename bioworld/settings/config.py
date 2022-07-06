from settings import default_settings

def getConfig(name,settings):
    config = {
        "default":default_settings.getSettings(settings) ,
    }
    return config.get(name, False) 
    

    