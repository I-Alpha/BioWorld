import time
import numpy as np
from math import floor
from math import sqrt
from math import degrees
from math import atan2


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    return "{0}:{1}:{2:0.2f}".format(int(hours), int(mins), sec)


def info_box(env_app, n_intervals):
    return [
        'time: {}'.format(time_convert(time.time() - env_app.time)),
        'gen_time: {}'.format(time_convert(env_app.gen_curr_time)),
        'live_organisms: {0:.2f}'.format(len(env_app.organisms)),
        'dead_organisms: {0:.2f}'.format(env_app.dead_organisms),
        'time_step: {0:0.2f}/{1}'.format(env_app.t_step,
                                         env_app.total_time_steps),
        'total: {0:0.2f}'.format(len(env_app.foods)+len(env_app.organisms)),
        'gen: {0:0.2f}'.format(env_app.gen),
        'food_len: {0:0.2f}'.format(len(env_app.foods)),
        'dash_intervals: {0:.2f}'.format(n_intervals),
    ]


def dist(x1, y1, x2, y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)


def calc_heading(org, food):
    d_x = food.x - org.x
    d_y = food.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180:
        theta_d += 360
    return theta_d / 180
