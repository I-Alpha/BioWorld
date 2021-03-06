from __future__ import division, print_function
from collections import defaultdict

from math import floor
import numbers
from random import randint
from random import random as rand
from random import sample
from random import uniform
from itertools import groupby
from tokenize import Number

from numpy import array
import numpy


from entities.food import *
from entities.agent import *

import operator


def evolve(settings, organisms_old, gen):
    organisms_new = []
    for key, group in groupby(organisms_old, lambda x: x.behaviour):
        organism_group = [agent for agent in group]
        if len(organism_group) < 2:
            return False, False
        s = int(floor(settings['elitism'] * len(organism_group)))
        elitism_num = s if s >= 2 else 2
        new_orgs = settings['agents'][key] - elitism_num

        # --- GET STATS FROM CURRENT GENERATION ----------------+
        stats = defaultdict(int)
        for org in organism_group:
            if org.fitness > stats['BEST'] or stats['BEST'] == 0:
                stats['BEST'] = org.fitness

            if org.fitness < stats['WORST'] or stats['WORST'] == 0:
                stats['WORST'] = org.fitness

            stats['SUM'] += org.fitness
            stats['COUNT'] += 1

        stats['AVG'] = stats['SUM'] / stats['COUNT']

        # --- ELITISM (KEEP BEST PERFORMING ORGANISMS) ---------+
        orgs_sorted = sorted(
            organism_group, key=operator.attrgetter('fitness'), reverse=True)
        for i in range(0, elitism_num):
            organisms_new.append(Agent(
                settings,
                wih=orgs_sorted[i].wih,
                who=orgs_sorted[i].who,
                neuralnet=orgs_sorted[i].brain,
                name=orgs_sorted[i].name,
                behaviour=key,
                mind_type=orgs_sorted[i].mind_type
            ))

        # --- GENERATE NEW ORGANISMS ---------------------------+
        for w in range(0, new_orgs):
            # SELECTION (TRUNCATION SELECTION)
            canidates = range(0, elitism_num)
            random_index = sample(canidates, 2)
            org_1 = orgs_sorted[random_index[0]]
            org_2 = orgs_sorted[random_index[1]]

            # CROSSOVER

            crossover_weight = rand()
            if settings['mind_type'] == "default":
                wih_new = (crossover_weight * org_1.wih) + \
                    ((1 - crossover_weight) * org_2.wih)
                who_new = (crossover_weight * org_1.who) + \
                    ((1 - crossover_weight) * org_2.who)
                # MUTATION
                mutate = rand()
                if mutate <= settings['mutate']:

                    # PICK WHICH WEIGHT MATRIX TO MUTATE
                    mat_pick = randint(0, 1)

                    # MUTATE: WIH WEIGHTS
                    if mat_pick == 0:
                        index_row = randint(0, settings['hnodes']-1)
                        wih_new[index_row] = wih_new[index_row] * \
                            uniform(0.9, 1.1)
                        for col in wih_new[index_row]:
                            if col > 1:
                                col = 1
                            if col < -1:
                                col = -1

                    # MUTATE: WHO WEIGHTS
                    if mat_pick == 1:
                        index_row = randint(0, settings['onodes']-1)
                        index_col = randint(0, settings['hnodes']-1)
                        who_new[index_row][index_col] = who_new[index_row][index_col] * \
                            uniform(0.9, 1.1)
                        if who_new[index_row][index_col] > 1:
                            who_new[index_row][index_col] = 1
                        if who_new[index_row][index_col] < -1:
                            who_new[index_row][index_col] = -1

                organisms_new.append(Agent(
                    settings, wih=wih_new, who=who_new, name='gen['+str(gen)+']-org['+str(w)+']'+key, behaviour=key))
            else:
                o1_w = org_1.brain.get_weights()
                o2_w = org_2.brain.get_weights()
                w_new = np.dot(crossover_weight, o1_w) + \
                    np.dot((1 - crossover_weight), o2_w)
                # MUTATION
                mutate = rand()
                if mutate <= settings['mutate']:
                    # PICK WHICH WEIGHT MATRIX TO MUTATE
                    mat_pick = randint(0, len(w_new)-1)

                    # MUTATE: WIH WEIGHTS
                    if mat_pick == 0:
                        w_new_row = randint(0, len(w_new[0])-1) 
                        w_new_col = randint(0, len(w_new[0][w_new_row])-1) 
                        w_new[0][w_new_row][w_new_col] = w_new[0][w_new_row][w_new_col] * \
                            uniform(0.9, 1.1) 
                        if w_new[0][w_new_row][w_new_col] > 1:
                            w_new[0][w_new_row][w_new_col] = 1
                        if w_new[0][w_new_row][w_new_col] < -1:
                            w_new[0][w_new_row][w_new_col] = -1

                    # MUTATE: WHO WEIGHTS
                    else:
                        shape = w_new.shape  
                        if mat_pick == 1 or mat_pick == 3:
                            index_row = 0
                            w_new[mat_pick][index_row]  = w_new[mat_pick][index_row] * \
                            uniform(0.9, 1.1)
                            if  w_new[mat_pick][index_row]  > 1:
                                w_new[mat_pick][index_row]  = 1
                            if w_new[mat_pick][index_row]  < -1:
                                w_new[mat_pick][index_row]  = -1
                            
                        if mat_pick == 2:
                            index_row = randint(0,len(w_new[mat_pick])-1)
                            index_col = randint(0,len(w_new[mat_pick][index_row])-1)
                            w_new[mat_pick][index_row][index_col]  = w_new[mat_pick][index_row][index_col]  * \
                            uniform(0.9, 1.1) 
                            if w_new[mat_pick][index_row][index_col]> 1:
                                w_new[mat_pick][index_row][index_col] = 1
                            if w_new[mat_pick][index_row][index_col] < -1:
                                w_new[mat_pick][index_row][index_col] = -1

                            
                        if mat_pick == 3:
                            index_row = randint(0,len(w_new[mat_pick])-1) 
                            w_new[mat_pick][index_row]  = w_new[mat_pick][index_row]  * \
                            uniform(0.9, 1.1)
                            if w_new[mat_pick][index_row]> 1:
                                w_new[mat_pick][index_row] = 1
                            if w_new[mat_pick][index_row] < -1:
                                w_new[mat_pick][index_row] = -1

                organisms_new.append(Agent(
                    settings, wih=[], who=[], w_new=w_new, name='gen['+str(gen)+']-org['+str(w)+']'+key, behaviour=key, mind_type="custom"))

    return organisms_new, stats
