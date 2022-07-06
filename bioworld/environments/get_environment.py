from environments import basic_env, default_env

def get_env(env_name):
    environments = {
        "default": default_env.Env,
        "basic": basic_env.Env,
    }
    return environments.get(env_name,False)
