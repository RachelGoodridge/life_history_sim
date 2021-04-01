###### Vectorized Version by RG ######
import numpy as np
import pandas as pd
import copy
import pdb
import pickle
from scipy import stats
import math
import os
import sys


# Food Growth
def grow_food(grid, var, grid_dim):
    grow_rate = np.random.normal(var["food_growth"][0], var["food_growth"][1])
    food_max = grid[:,:,grid_dim["food"]] - (grid[:,:,grid_dim["food"]] % (-1*var["food_max"]))
    food_max[food_max < var["food_max"]] = var["food_max"]
    grid[:,:,grid_dim["food"]] = (food_max*grid[:,:,grid_dim["food"]]*np.exp(grow_rate))/(food_max+(grid[:,:,grid_dim["food"]]*np.expm1(grow_rate)))
    # random repopulation of a food patch at some oscillating rate
    repop_rate = var["food_amp"] * math.sin(var["food_freq"] * var["iter"]) + var["food_repop"]
    if np.random.choice([0,1], p=[(1-repop_rate),repop_rate]):
        corners = [i for i in range(0, var["grid_len"], var["food_len"] + var["space_between"])]
        i,j = np.random.choice(corners,size=2)
        grid[:,:,grid_dim["food"]][i:(i + var["food_len"]), j:(j + var["food_len"])] += var["food_start"]
    
# Pheromone Decay
def decay_pheromones(grid, var, grid_dim):
    grid[:,:,grid_dim["pher"]] = grid[:,:,grid_dim["pher"]]*np.exp(var["pher_decay"])
    grid[:,:,grid_dim["pher"]][grid[:,:,grid_dim["pher"]]<0] = 0

# Update Worm Locations
def update_grid(grid, var, stage, grid_dim, p2i, df):
    df2 = df[df[:,p2i["alive"]].astype(bool)]
    df2 = df2[:,[p2i["x_loc"],p2i["y_loc"],p2i["stage"],p2i["gender"]]]
    # stretch out properties based on unique x, y, stage, and gender
    factor = np.array([1,var["grid_len"],var["grid_len"]**2,(var["grid_len"]**2)*len(stage)])
    position = df2.dot(factor).astype(int)
    grid2 = np.zeros((var["grid_len"],var["grid_len"],len(stage)*2), order = "F")
    # find overlaps in all properties
    unique, count = np.unique(position, return_counts = True)
    grid2[np.unravel_index(unique, grid2.shape, order = "F")] = count
    grid[:,:,grid_dim["f_egg"]:] = grid2

# Update Mates Array
def list_sperm(mates, s2i, prop, df, p2i):
    # only keep the males and females that are alive
    sorter = np.argsort(df[:,p2i["name"]].astype(int))
    male_index = sorter[np.searchsorted(df[:,p2i["name"]].astype(int), mates[:,1], sorter=sorter)]
    mates = mates[df[male_index,p2i["alive"]]==1]
    female_index = sorter[np.searchsorted(df[:,p2i["name"]].astype(int), mates[:,0], sorter=sorter)]
    mates = mates[df[female_index,p2i["alive"]]==1]   
    # find all adult worms
    df2 = pd.DataFrame(df, columns = prop)
    df2 = df2.loc[df2["alive"]==1]
    df2 = df2.loc[df2["stage"]==s2i["adult"],["name","x_loc","y_loc","gender"]]
    # separate into male and female
    males = df2.loc[df2["gender"]==1]
    females = df2.loc[df2["gender"]==0]
    # find which males on the same x and y locations as females
    location = females.merge(males, on = ["x_loc","y_loc"])
    who = location[["name_x","name_y"]]
    # add to the mates array
    mates = np.concatenate((mates,who.to_numpy()))
    return(mates)

# Update Neighbors
def update_nbrs(grid, north, south, west, east, var):
    # North in xy coordinates is West in python arrays
    north[:, :(var["grid_len"]-1), :] = grid[:, 1:var["grid_len"], :]
    north[:, (var["grid_len"]-1), :] = grid[:, 0, :]
    # South in xy coordinates is East in python arrays
    south[:, 1:var["grid_len"], :] = grid[:, :(var["grid_len"]-1), :]
    south[:, 0, :] = grid[:, (var["grid_len"]-1), :]
    # West in xy coordinates is South in python arrays
    west[1:var["grid_len"], :, :] = grid[:(var["grid_len"]-1), :, :]
    west[0, :, :] = grid[(var["grid_len"]-1), :, :]
    # East in xy coordinates is North in python arrays
    east[:(var["grid_len"]-1), :, :] = grid[1:var["grid_len"], :, :]
    east[(var["grid_len"]-1), :, :] = grid[0, :, :]   

# Subtract Energy from Worms
def metabolism(df, s2i, p2i, var):
    # find all worms that are alive
    every = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    # kill off dauer worms after 4 months
    dauer = every[df[every,p2i["stage"]] == s2i["dauer"]]
    to_die = dauer[df[dauer,p2i["dauer"]] > var["dauer_age"]]
    if len(to_die) > 0:
        df[to_die,p2i["alive"]] = 0
        df[to_die,p2i["decision"]] = 0
        # find all worms that are alive
        every = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    # find their metabolism amount
    stages = df[every,p2i["stage"]].astype(int)
    energy_needed = np.array(var["energy_used"])[stages]
    # find bugs - specifically looking for worms with negative energy    
    if np.any(df[every,p2i["energy"]]<0):
        print("There are worms with negative energy.")
        pdb.set_trace()
    # remove the worms that will starve
    dead = df[every,p2i["energy"]] < energy_needed
    df[every[dead],p2i["alive"]] = 0
    df[every[dead],p2i["decision"]] = 0
    # subtract the energy needed for maintenance
    which_worm = df[every,p2i["energy"]] >= energy_needed
    df[every[which_worm],p2i["energy"]] -= energy_needed[which_worm]


# Move Chosen Worms
def move(grid, df, var, s2i, grid_dim, north, south, west, east, p2i):
    # find worms alive and decision True
    every = np.array(np.where((df[:,p2i["alive"]]==1) & (df[:,p2i["decision"]]==1)))[0]
    
    # find surrounding pheromones
    p_list = [north[:,:,grid_dim["pher"]], south[:,:,grid_dim["pher"]], west[:,:,grid_dim["pher"]], east[:,:,grid_dim["pher"]]]
    p_sum = (1/(1+p_list[0])) + (1/(1+p_list[1])) + (1/(1+p_list[2])) + (1/(1+p_list[3]))
    p_sum_m = p_list[0] + p_list[1] + p_list[2] + p_list[3] + 4
    # create probability of moving dictionaries based on fraction of pher in each direction
    p_prob = {"N":((1/(1+p_list[0]))/p_sum), "S":((1/(1+p_list[1]))/p_sum), "W":((1/(1+p_list[2]))/p_sum), "E":((1/(1+p_list[3]))/p_sum)}
    p_prob_m = {"N":((p_list[0]+1)/p_sum_m), "S":((p_list[1]+1)/p_sum_m), "W":((p_list[2]+1)/p_sum_m), "E":((p_list[3]+1)/p_sum_m)}
    
    # find surrounding food
    f_list = [north[:,:,grid_dim["food"]], south[:,:,grid_dim["food"]], west[:,:,grid_dim["food"]], east[:,:,grid_dim["food"]]]
    f_sum = f_list[0] + f_list[1] + f_list[2] + f_list[3] + 4
    # create probability of moving dictionaries based on fraction of food in each direction
    f_prob = {"N":((f_list[0]+1)/f_sum), "S":((f_list[1]+1)/f_sum), "W":((f_list[2]+1)/f_sum), "E":((f_list[3]+1)/f_sum)}
    
    # find adult males and all others
    adult_male = every[((df[every,p2i["stage"]]==s2i["adult"]) & (df[every,p2i["gender"]]==1))]
    other = every[~((df[every,p2i["stage"]]==s2i["adult"]) & (df[every,p2i["gender"]]==1))]
    
    # variables for adult males
    move_var_m = []
    if len(adult_male)>0:
        # x and y locations, as well as smell genes
        move_var_m = np.c_[adult_male, df[adult_male,p2i["x_loc"]], df[adult_male,p2i["y_loc"]], (df[adult_male,p2i["smell_1"]] + df[adult_male,p2i["smell_2"]])/2]
        for d in ["N", "S", "W", "E"]:
            # food and pher probabilities in the shape of the grid
            move_var_m = np.c_[move_var_m, f_prob[d][np.unravel_index(df[adult_male,p2i["loc"]].astype(int), f_prob[d].shape, order = "F")]]
            move_var_m = np.c_[move_var_m, p_prob_m[d][np.unravel_index(df[adult_male,p2i["loc"]].astype(int), p_prob_m[d].shape, order = "F")]]
    
    # variables for all others
    move_var = []
    if len(other)>0:        
        # x and y locations, as well as smell genes
        move_var = np.c_[other, df[other,p2i["x_loc"]], df[other,p2i["y_loc"]], (df[other,p2i["smell_1"]] + df[other,p2i["smell_2"]])/2]
        for d in ["N", "S", "W", "E"]:
            # food and pher probabilities in the shape of the grid
            move_var = np.c_[move_var, f_prob[d][np.unravel_index(df[other,p2i["loc"]].astype(int), f_prob[d].shape, order = "F")]]
            move_var = np.c_[move_var, p_prob[d][np.unravel_index(df[other,p2i["loc"]].astype(int), p_prob[d].shape, order = "F")]]
    
    # combine variables for adult males and all others
    move_all = np.zeros((len(move_var_m) + len(move_var), 12))
    if len(move_all) > 0:
        if len(move_var_m) > 0:
            move_all[:len(move_var_m),:] = move_var_m
        if len(move_var) > 0:
            move_all[-(len(move_var)):,:] = move_var    
    
    # determine probabilities and new locations
        # smell gene * food probs + (1-smell gene) * pher probs
        move_prob = move_all[:,3][:,np.newaxis]*move_all[:,4::2] + (1-move_all[:,3][:,np.newaxis])*move_all[:,5::2]
        moves = np.array([[0,1], [0,-1], [-1,0], [1,0]])
        # choice of moves based on move_prob applied to all worms
        movement = np.apply_along_axis(lambda row: np.random.choice(len(moves), p = row), 1, move_prob)
        # adjust x and y separately, making sure they don't fall off the grid
        new_xloc = ((move_all[:,1] + moves[movement][:,0]) % var["grid_len"]).astype(int)
        new_yloc = ((move_all[:,2] + moves[movement][:,1]) % var["grid_len"]).astype(int)
        new_loc = np.ravel_multi_index([new_xloc, new_yloc], grid.shape[:2], order = "F")
        df[move_all[:,0].astype(int),p2i["x_loc"]] = new_xloc
        df[move_all[:,0].astype(int),p2i["y_loc"]] = new_yloc
        df[move_all[:,0].astype(int),p2i["loc"]] = new_loc


# Worms Leave Pheromones
def deposit_pher(grid, var, s2i, grid_dim, p2i, df):
    every = np.array(np.where((df[:,p2i["alive"]]==1) & (df[:,p2i["stage"]]!=s2i["egg"])))[0]
    # pher amount based on stage and placed in the correct location
    pher_amt = np.bincount(df[every,p2i["loc"]].astype(int), weights = np.array(var["pher"])[df[every,p2i["stage"]].astype(int)])
    new_pher = np.zeros(grid[:,:,1].shape, order = "F")
    # fit pher amounts to grid shape and add to main grid
    new_pher[np.unravel_index(np.arange(len(pher_amt)), new_pher.shape, order = "F")] = pher_amt
    grid[:,:,grid_dim["pher"]] += new_pher


# Worms Advance Stages
def grow(df, var, s2i, p2i, grid, grid_dim, north, south, west, east):
    # transition to next stage for all except L1, L2d, dauer, and old
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    # pick out the worms that are in pick_stage
    pick_stage = [s2i["egg"], s2i["L2"], s2i["L3"], s2i["L4"], s2i["adult"]]
    worms = alive[np.isin(df[alive,p2i["stage"]], pick_stage)]
    # determine if food_count reaches or exceeds the grow_goal for that specific stage
    grow_goal = np.array(var["grow_time"])[df[worms,p2i["stage"]].astype(int)]
    worms = worms[df[worms,p2i["food_count"]] >= grow_goal]
    L2_L3 = worms[(df[worms,p2i["stage"]]==s2i["L2"]) | (df[worms,p2i["stage"]]==s2i["L3"])]
    other = worms[~((df[worms,p2i["stage"]]==s2i["L2"]) | (df[worms,p2i["stage"]]==s2i["L3"]))]
    # transition to the next stage (must add 2 for L2 and L3 because of order of stages)
    df[L2_L3, p2i["stage"]] += 2
    df[other, p2i["stage"]] += 1
    # if energy is more than certain amount, reset it
    stages = df[worms,p2i["stage"]].astype(int)
    energy_amt = var["energy"]*np.array(var["energy_used"])[stages]
    which_worm = df[worms,p2i["energy"]] > energy_amt
    df[worms[which_worm],p2i["energy"]] = energy_amt[which_worm]
    
    # transition from L1 -> L2 or L2d
    L1 = alive[((df[alive,p2i["stage"]]==s2i["L1"]) & (df[alive,p2i["food_count"]]>=var["grow_time"][s2i["L1"]]))]
    if len(L1) > 0:
        # pull out gene values
        dauer_val = (df[L1,p2i["dauer_1"]] + df[L1,p2i["dauer_2"]])/2
        # find surrounding pheromones
        phers = north[:,:,grid_dim["pher"]] + south[:,:,grid_dim["pher"]] + west[:,:,grid_dim["pher"]] + east[:,:,grid_dim["pher"]] + grid[:,:,grid_dim["pher"]]
        pher_loc = np.ravel(phers, order="F")[df[L1, p2i["loc"]].astype(int)]
        pher_loc[pher_loc > var["pher_max"]] = var["pher_max"]
        # calculate probability based on dauer gene and time spent in L1
        prob = 1/(1 + np.exp((dauer_val - df[L1,p2i["L1"]])/var["gp_map"]))
        # 0.5*above prob + 0.5*pheromone fraction
        new_prob = 0.5*prob + 0.5*pher_loc/var["pher_max"]
        probs = np.c_[new_prob,1-new_prob]
        # choice of L2 or L2d based on probs, applied to all worms
        new_stage = np.apply_along_axis(lambda x:np.random.choice([s2i["L2d"],s2i["L2"]],p=x),1,probs)
        df[L1,p2i["stage"]] = new_stage
        # if energy is more than certain amount, reset it
        L1 = L1[df[L1,p2i["energy"]] > (var["energy"]*var["energy_used"][s2i["L2"]])]
        df[L1,p2i["energy"]] = var["energy"]*var["energy_used"][s2i["L2"]]
    
    # transition from L2d -> L3 or dauer
    L2d = alive[((df[alive,p2i["stage"]]==s2i["L2d"]) & (df[alive,p2i["food_count"]]>=var["grow_time"][s2i["L2d"]]))]
    if len(L2d) > 0:
        # pull out gene values
        dauer_val = (df[L2d,p2i["dauer_1"]] + df[L2d,p2i["dauer_2"]])/2
        # find surrounding pheromones
        phers = north[:,:,grid_dim["pher"]] + south[:,:,grid_dim["pher"]] + west[:,:,grid_dim["pher"]] + east[:,:,grid_dim["pher"]] + grid[:,:,grid_dim["pher"]]
        pher_loc = np.ravel(phers, order="F")[df[L2d, p2i["loc"]].astype(int)]
        pher_loc[pher_loc > var["pher_max"]] = var["pher_max"]
        # calculate probability based on dauer gene and time spent in L2d
        prob = 1/(1 + np.exp((dauer_val - df[L2d,p2i["L2d"]])/var["gp_map"]))
        # 0.5*above prob + 0.5*pheromone fraction
        new_prob = 0.5*prob + 0.5*pher_loc/var["pher_max"]
        probs = np.c_[new_prob,1-new_prob]
        # choice of L3 or dauer based on probs, applied to all worms
        new_stage = np.apply_along_axis(lambda x:np.random.choice([s2i["dauer"],s2i["L3"]],p=x),1,probs)
        df[L2d,p2i["stage"]] = new_stage
        df[L2d,p2i["food_count"]] = var["grow_time"][s2i["L2"]]
        # if energy is more than certain amount, reset it
        reset = L2d[df[L2d,p2i["energy"]] > (var["energy"]*var["energy_used"][s2i["L3"]])]
        df[reset,p2i["energy"]] = var["energy"]*var["energy_used"][s2i["L3"]]
        # find the ones that went into dauer and determine if they will survive
        new_dauer = L2d[df[L2d,p2i["stage"]]==s2i["dauer"]]
        to_die = np.random.choice([True,False], p=(var["dauer_die"],1-var["dauer_die"]), size=len(new_dauer))
        new_dauer = new_dauer[to_die]
        if len(new_dauer) > 0:
            df[new_dauer,p2i["dauer"]] = var["dauer_age"]
            df[new_dauer,p2i["alive"]] = 0
            df[new_dauer,p2i["decision"]] = 0
    
    # cutoff L2d worms that have not eaten enough food to transition
    L2d = alive[(df[alive,p2i["stage"]]==s2i["L2d"])]
    if len(L2d) > 0:
        # worms that have spent too long in L2d must go into dauer
        dauer_val = (df[L2d,p2i["dauer_1"]] + df[L2d,p2i["dauer_2"]])/2
        which_one = L2d[((1/(1 + np.exp((dauer_val - df[L2d,p2i["L2d"]])/var["gp_map"]))) >= var["L2d_cutoff"])]
        df[which_one,p2i["stage"]] = s2i["dauer"]
        # reset the food_count and the energy
        df[which_one,p2i["food_count"]] = var["grow_time"][s2i["L2"]]
        reset = which_one[df[which_one,p2i["energy"]] > (var["energy"]*var["energy_used"][s2i["L3"]])]
        df[reset,p2i["energy"]] = var["energy"]*var["energy_used"][s2i["L3"]]
        # determine if the dauer worms will survive
        to_die = np.random.choice([True,False], p=(var["dauer_die"],1-var["dauer_die"]), size=len(which_one))
        which_one = which_one[to_die]
        if len(which_one) > 0:
            df[which_one,p2i["dauer"]] = var["dauer_age"]
            df[which_one,p2i["alive"]] = 0
            df[which_one,p2i["decision"]] = 0
    
    # choice to come out of dauer or not
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    dauer = alive[((df[alive,p2i["stage"]]==s2i["dauer"]) & (df[alive,p2i["food_count"]]>=var["grow_time"][s2i["dauer"]]))]
    if len(dauer) > 0:
        # find food at dauer location
        food = grid[:,:,grid_dim["food"]]
        food_loc = np.ravel(food, order="F")[df[dauer, p2i["loc"]].astype(int)]
        food_loc[food_loc > var["food_max"]] = var["food_max"]
        # find surrounding pheromones
        phers = north[:,:,grid_dim["pher"]] + south[:,:,grid_dim["pher"]] + west[:,:,grid_dim["pher"]] + east[:,:,grid_dim["pher"]] + grid[:,:,grid_dim["pher"]]
        pher_loc = np.ravel(phers, order="F")[df[dauer, p2i["loc"]].astype(int)]
        pher_loc[pher_loc > var["pher_max"]] = var["pher_max"]
        # chance of staying in dauer is 0.5*(1-food) + 0.5*pher
        dauer_prob = 0.5*(1-food_loc/var["food_max"]) + 0.5*pher_loc/var["pher_max"]
        probs = np.c_[dauer_prob, 1-dauer_prob]
        # choice of staying in dauer or coming out based on probs, applied to all worms
        new_stage = np.apply_along_axis(lambda x: np.random.choice([s2i["dauer"],s2i["L4"]],p=x),1,probs)
        df[dauer,p2i["stage"]] = new_stage
        # find all the ones that transitioned out of dauer and reset food count
        dauer = dauer[df[dauer,p2i["stage"]]==s2i["L4"]]
        df[dauer,p2i["food_count"]] = var["grow_time"][s2i["L3"]]
        # if energy is more than certain amount, reset it
        dauer = dauer[df[dauer,p2i["energy"]] > (var["energy"]*var["energy_used"][s2i["L4"]])]
        df[dauer,p2i["energy"]] = var["energy"]*var["energy_used"][s2i["L4"]]
        
    # death after old age
    dead = alive[((df[alive,p2i["stage"]]==s2i["old"]) & (df[alive,p2i["food_count"]]>=var["grow_time"][s2i["old"]]))]
    if len(dead) > 0:
        df[dead,p2i["alive"]] = 0
        df[dead,p2i["decision"]] = 0


# Worms Eat Food
def eat(df, grid, var, s2i, grid_dim, p2i):
    # add food count to eggs
    egg = np.array(np.where((df[:,p2i["alive"]]==1) & (df[:,p2i["stage"]]==s2i["egg"])))[0]
    df[egg, p2i["food_count"]] += 1
    
    # add food count to dauer
    dauer = np.array(np.where((df[:,p2i["alive"]]==1) & (df[:,p2i["stage"]]==s2i["dauer"])))[0]
    # find locations and food at each
    locs = df[dauer, p2i["loc"]].astype(int)
    food = np.ravel(grid[:,:,grid_dim["food"]], order = "F")
    food_loc = food[locs]
    # find how many worms share the same location
    alive = (df[:,p2i["alive"]]==1) & (df[:,p2i["stage"]]!=s2i["egg"])
    all_locs = df[alive,p2i["loc"]].astype(int)
    all_nbrs = np.bincount(all_locs)[locs]
    # add the minimum of either food at loc or food portion for dauer
    f_minimum = np.array([var["food_eaten"][s2i["dauer"]]]*len(dauer))
    df[dauer, p2i["food_count"]] += np.minimum(f_minimum, (food_loc/all_nbrs))
    
    # get list of food portions and locations for living worms, not eggs or dauer
    alive = np.array(np.where((df[:,p2i["alive"]]==1) & (df[:,p2i["stage"]]!=s2i["egg"]) & (df[:,p2i["stage"]]!=s2i["dauer"])))[0]
    portion = np.array(var["food_eaten"])[df[alive, p2i["stage"]].astype(int)]
    # determine locations and food where worms are
    locs = df[alive, p2i["loc"]].astype(int)
    food_loc = food[locs]
    portion[food_loc == 0] = 0
    
    # determine how much should be eaten in each location
    eat_amt = np.bincount(locs, weights = portion)
    eat_loc = np.arange(len(eat_amt))
    food_loc = food[eat_loc]
    # determine how much would be overeaten in each location
    over_eat = (eat_amt - food_loc)*(eat_amt > food_loc)
    eat_amt[eat_amt == 0] = 1
    scale = 1 - over_eat/eat_amt
    
    # add energy to each worm based on their portion and scale
    energy = scale[locs]*portion
    df[alive, p2i["energy"]] += energy
    df[alive, p2i["food_count"]] += energy
    
    # remove food from the grid
    eat_amt = np.bincount(locs, weights = portion)
    eaten = eat_amt - over_eat
    grid[:,:,grid_dim["food"]][np.unravel_index(eat_loc, grid[:,:,grid_dim["food"]].shape, order = "F")] -= eaten
    grid[:,:,grid_dim["food"]][grid[:,:,grid_dim["food"]]<0] = 0
    

# Worms Reproduce
def reproduce(df, mates, var, s2i, g2i, prop, p2i):
    # find all the adult females/herms with energy
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    adult_f = alive[((df[alive,p2i["gender"]]==g2i["herm"]) & (df[alive,p2i["stage"]]==s2i["adult"]) & (df[alive,p2i["energy"]]>=1))]
    # sort all females/herms into with or without sperm categories
    with_sperm = np.array(np.where(np.isin(df[:,p2i["name"]], mates[:,0].astype(int))))[0]
    with_sperm = with_sperm[np.isin(with_sperm, adult_f)]
    choose = np.random.choice([1,0], size=len(with_sperm), p=[var["sperm_bias"],(1-var["sperm_bias"])])
    with_sperm = with_sperm[choose==1]
    without_sperm = adult_f[~np.isin(adult_f, with_sperm)]
    # create the following local variables as a placeholder
    num_eggs_without = []
    num_eggs_with = []
    
    if (var["gender_prob"] == var["genders_prob"][0]) and (len(without_sperm) > 0):
        # each herm will randomly choose to lay an egg or not
        num_eggs_without = np.random.choice([1,0], size=len(without_sperm), p=[var["eggs"],(1-var["eggs"])])
        df[without_sperm, p2i["energy"]] -= num_eggs_without
        # which worms will have an egg
        without_sperm = without_sperm[num_eggs_without>0]
        num_eggs_without = num_eggs_without[num_eggs_without>0]
        
    if (len(with_sperm) > 0):
        # each female/herm will randomly choose to lay an egg or not
        num_eggs_with = np.random.choice([1,0], size=len(with_sperm), p=[var["eggs"],(1-var["eggs"])])
        df[with_sperm, p2i["energy"]] -= num_eggs_with
        # which worms will have an egg
        with_sperm = with_sperm[num_eggs_with>0]
        num_eggs_with = num_eggs_with[num_eggs_with>0]
        
    # find the start and end points in the dataframe
    try:
        first_nan = np.min(np.where(np.isnan(df[:, p2i["gender"]])))
    except:
        first_nan = df.shape[0]
    last_nan = first_nan + np.sum(num_eggs_with) + np.sum(num_eggs_without)
    # if the dataframe is too small, then create a new one and save the old one
    if (last_nan > df.shape[0]):
        # remember the names of the worms that will lay eggs
        if (len(with_sperm) > 0):
            with_sperm_name = df[with_sperm, p2i["name"]].astype(int)
        if (len(without_sperm) > 0):
            without_sperm_name = df[without_sperm, p2i["name"]].astype(int)
        
        # seaparte all existing worms into alive and dead
        subset = df[~np.isnan(df[:,p2i["gender"]]),:]
        alive = subset[subset[:,p2i["alive"]]==1,:]
        dead = subset[subset[:,p2i["alive"]]==0,:]
        
        # overwrite array and put living worms in there first
        df[:len(alive),:] = alive
        df[len(alive):,:] = np.nan*np.zeros((var["pop_max"]-len(alive),alive.shape[1]),order="F")
        # figure out the names of the worms in the new array
        first_worm = subset[-1,p2i["name"]].astype(int) + 1
        last_worm = first_worm + var["pop_max"] - len(alive)
        # preset the properties of the future worms
        df[len(alive):,p2i["name"]] = np.arange(first_worm, last_worm)
        df[len(alive):,p2i["food_count"]] = 0
        df[len(alive):,p2i["stage"]] = 0
        df[len(alive):,p2i["energy"]] = var["energy"]*var["energy_used"][s2i["L1"]]
        df[len(alive):,p2i["L1"]:p2i["gene_0"]] = 0
        
        # save the dead worms in a file for later
        file_name = "dead_worm_data_dump_" + str(var["data"]) + ".p"
        new_file = open(file_name, "wb")
        pickle.dump(dead, new_file)
        new_file.close()
        print("Dead Worm Data Dump " + str(var["data"]))
        var["data"] += 1
        
        # get new indices for the worms that will lay eggs
        if (len(with_sperm) > 0):
            with_sperm = np.array(np.where(np.isin(df[:,p2i["name"]], with_sperm_name)))[0]
        if (len(without_sperm) > 0):
            without_sperm = np.array(np.where(np.isin(df[:,p2i["name"]], without_sperm_name)))[0]
        
    # self fertilizate only if hermaphrodite with no sperm: verify androdioecy
    if (var["gender_prob"] == var["genders_prob"][0]) and (len(without_sperm) > 0):            
        # find the start and end points in the dataframe
        first_nan = np.min(np.where(np.isnan(df[:, p2i["gender"]])))
        last_nan = first_nan + np.sum(num_eggs_without)
        
        # fill in properties for the new worms based on parent information
        df[first_nan:last_nan, p2i["gender"]] = np.random.choice(var["gender"], p=var["gender_prob"], size=np.sum(num_eggs_without))
        df[first_nan:last_nan, p2i["x_loc"]] = df[without_sperm, p2i["x_loc"]]
        df[first_nan:last_nan, p2i["y_loc"]] = df[without_sperm, p2i["y_loc"]]
        df[first_nan:last_nan, p2i["loc"]] = df[without_sperm, p2i["loc"]]
        df[first_nan:last_nan, p2i["parent_1"]] = df[without_sperm, p2i["name"]]
        df[first_nan:last_nan, p2i["parent_2"]] = df[without_sperm, p2i["name"]]
        df[first_nan:last_nan, p2i["alive"]] = 1
        
        # fill out dauer gene with possible mutation
        dauer_1_gene = df[without_sperm, p2i["dauer_1"]]
        dauer_2_gene = df[without_sperm, p2i["dauer_2"]]
        mutation = np.random.choice([0,1],size=(np.sum(num_eggs_without),2),p=[1-var["mutation_rate"],var["mutation_rate"]])*np.random.normal(0,var["dauer_weight"],size=(np.sum(num_eggs_without),2))
        # pick dauer_1 or dauer_2 and add the mutated amount
        df[first_nan:last_nan, p2i["dauer_1"]] = np.apply_along_axis(np.random.choice, 0, [dauer_1_gene,dauer_2_gene]) + mutation[:,0]
        df[first_nan:last_nan, p2i["dauer_2"]] = np.apply_along_axis(np.random.choice, 0, [dauer_1_gene,dauer_2_gene]) + mutation[:,1]
        
        # fill out smell gene with possible mutation
        smell_1_gene = df[without_sperm, p2i["smell_1"]]
        smell_2_gene = df[without_sperm, p2i["smell_2"]]
        mutation = np.random.choice([0,1],size=(np.sum(num_eggs_without),2),p=[1-var["mutation_rate"],var["mutation_rate"]])*np.random.normal(0,var["smell_weight"],size=(np.sum(num_eggs_without),2))
        # pick smell_1 or smell_2 and add the mutated amount - be sure it is between 0 and 1
        new_smell_1 = np.apply_along_axis(np.random.choice, 0, [smell_1_gene,smell_2_gene]) + mutation[:,0]
        new_smell_1[new_smell_1 > 1] = 1
        new_smell_1[new_smell_1 < 0] = 0
        new_smell_2 = np.apply_along_axis(np.random.choice, 0, [smell_1_gene,smell_2_gene]) + mutation[:,1]
        new_smell_2[new_smell_2 > 1] = 1
        new_smell_2[new_smell_2 < 0] = 0
        df[first_nan:last_nan, p2i["smell_1"]] = new_smell_1
        df[first_nan:last_nan, p2i["smell_2"]] = new_smell_2
        
        # fill out genetic information with choice of allele 1 or 2 for each gene
        num_genes = var["genes"]*2
        num_parents = len(without_sperm)
        parents = range(num_parents)
        parent_genes = df[without_sperm, p2i["gene_0"]:]
        # combine 3 matrices to choose allele based on parent allele indices: choose allele 1 or 2, which gene to select from, which parent to select from
        allele_choice = ((np.random.choice([0,1], size=(np.sum(num_eggs_without), num_genes))*num_parents) +
                         (np.repeat(np.arange(0,num_genes,2),np.sum(num_eggs_without)*2).reshape((np.sum(num_eggs_without),num_genes),order="F")*num_parents) +
                         (np.tile(parents,num_genes).reshape((np.sum(num_eggs_without),num_genes),order="F"))).astype(int)
        df[first_nan:last_nan, p2i["gene_0"]:] = parent_genes[np.unravel_index(allele_choice,parent_genes.shape,order="F")]
        
    # reproduce using sperm if either female or hermaphrodite
    if (len(with_sperm) > 0):
        # find the start and end points in the dataframe
        first_nan = np.min(np.where(np.isnan(df[:, p2i["gender"]])))
        last_nan = first_nan + np.sum(num_eggs_with)
        
        # fill in properties for the new worms based on parent information
        df[first_nan:last_nan, p2i["gender"]] = np.random.choice(var["gender"], p=var["genders_prob"][1], size=np.sum(num_eggs_with))
        df[first_nan:last_nan, p2i["x_loc"]] = df[with_sperm, p2i["x_loc"]]
        df[first_nan:last_nan, p2i["y_loc"]] = df[with_sperm, p2i["y_loc"]]
        df[first_nan:last_nan, p2i["loc"]] = df[with_sperm, p2i["loc"]]
        df[first_nan:last_nan, p2i["parent_1"]] = df[with_sperm, p2i["name"]]
        df[first_nan:last_nan, p2i["alive"]] = 1
        
        # each egg must choose a dad based on mom's sperm (only works for 1 egg per female)
        with_sperm_name = df[with_sperm,p2i["name"]].astype(int)
        pick_male = np.vectorize(lambda female: np.random.choice(mates[np.in1d(mates[:,0], female),1]))
        dad_name = np.apply_along_axis(pick_male, 0, with_sperm_name)
        df[first_nan:last_nan, p2i["parent_2"]] = dad_name
        
        # fill out dauer gene with possible mutation
        mom_dauer_1_gene = df[with_sperm, p2i["dauer_1"]]
        mom_dauer_2_gene = df[with_sperm, p2i["dauer_2"]]
        sorter = np.argsort(df[:,p2i["name"]].astype(int))
        dads = sorter[np.searchsorted(df[:,p2i["name"]].astype(int), dad_name, sorter=sorter)]
        dad_dauer_1_gene = df[dads, p2i["dauer_1"]]
        dad_dauer_2_gene = df[dads, p2i["dauer_2"]]
        mutation = np.random.choice([0,1],size=(np.sum(num_eggs_with),2),p=[1-var["mutation_rate"],var["mutation_rate"]])*np.random.normal(0,var["dauer_weight"],size=(np.sum(num_eggs_with),2))
        # pick dauer_1 or dauer_2 and add the mutated amount
        df[first_nan:last_nan, p2i["dauer_1"]] = np.apply_along_axis(np.random.choice, 0, [mom_dauer_1_gene,mom_dauer_2_gene]) + mutation[:,0]
        df[first_nan:last_nan, p2i["dauer_2"]] = np.apply_along_axis(np.random.choice, 0, [dad_dauer_1_gene,dad_dauer_2_gene]) + mutation[:,1]
        
        # fill out smell gene with possible mutation
        mom_smell_1_gene = df[with_sperm, p2i["smell_1"]]
        mom_smell_2_gene = df[with_sperm, p2i["smell_2"]]
        dad_smell_1_gene = df[dads, p2i["smell_1"]]
        dad_smell_2_gene = df[dads, p2i["smell_2"]]
        mutation = np.random.choice([0,1],size=(np.sum(num_eggs_with),2),p=[1-var["mutation_rate"],var["mutation_rate"]])*np.random.normal(0,var["smell_weight"],size=(np.sum(num_eggs_with),2))
        # pick smell_1 or smell_2 and add the mutated amount - be sure it is between 0 and 1
        new_smell_1 = np.apply_along_axis(np.random.choice, 0, [mom_smell_1_gene,mom_smell_2_gene]) + mutation[:,0]
        new_smell_1[new_smell_1 > 1] = 1
        new_smell_1[new_smell_1 < 0] = 0
        new_smell_2 = np.apply_along_axis(np.random.choice, 0, [dad_smell_1_gene,dad_smell_2_gene]) + mutation[:,1]
        new_smell_2[new_smell_2 > 1] = 1
        new_smell_2[new_smell_2 < 0] = 0
        df[first_nan:last_nan, p2i["smell_1"]] = new_smell_1
        df[first_nan:last_nan, p2i["smell_2"]] = new_smell_2        
        
        # fill out genetic information with choice of allele 1 or 2 for each gene
        num_genes = var["genes"]*2
        mom_genes = df[with_sperm, p2i["gene_0"]:]
        dads2 = np.unique(dads).astype(int)
        dad_genes = df[dads2, p2i["gene_0"]:]
        mom_index = range(len(with_sperm))
        sorter = np.argsort(dads2)
        dad_index = sorter[np.searchsorted(dads2, dads, sorter=sorter)]
        # combine 3 matrices to choose allele based on parent allele indices: choose allele 1 or 2, which gene to select from, which parent to select from
        mom_allele_choice = ((np.random.choice([0,1], size=(np.sum(num_eggs_with),var["genes"]))*len(with_sperm)) +
                             (np.repeat(np.arange(0,num_genes,2),np.sum(num_eggs_with)).reshape((np.sum(num_eggs_with),var["genes"]),order="F")*len(with_sperm)) +
                             (np.tile(mom_index, var["genes"]).reshape((np.sum(num_eggs_with),var["genes"]),order="F"))).astype(int)
        df[first_nan:last_nan, p2i["gene_0"]::2] = mom_genes[np.unravel_index(mom_allele_choice,mom_genes.shape,order="F")]
        dad_allele_choice = ((np.random.choice([0,1], size=(np.sum(num_eggs_with),var["genes"]))*len(dads2)) +
                             (np.repeat(np.arange(0,num_genes,2),np.sum(num_eggs_with)).reshape((np.sum(num_eggs_with),var["genes"]),order="F")*len(dads2)) +
                             (np.tile(dad_index,var["genes"]).reshape((np.sum(num_eggs_with),var["genes"]),order="F"))).astype(int)
        df[first_nan:last_nan, p2i["gene_1"]::2] = dad_genes[np.unravel_index(dad_allele_choice,dad_genes.shape,order="F")]
    return(df)


# Count Time Spent per Stage
def pass_time(df, s2i, p2i):
    not_egg = np.array(np.where((df[:,p2i["alive"]]==1) & (df[:,p2i["stage"]]!=s2i["egg"])))[0]
    # convert from stage index to property index
    stages_index = df[not_egg, p2i["stage"]].astype(int)
    property_index = stages_index + p2i["L1"] - 1
    df[not_egg, property_index] += 1


# Worms Decide to Travel
def decide(grid, df, var, s2i, grid_dim, north, south, west, east, p2i):
    # find all the food and surrounding pheromones
    phers = north[:,:,grid_dim["pher"]] + south[:,:,grid_dim["pher"]] + west[:,:,grid_dim["pher"]] + east[:,:,grid_dim["pher"]] + grid[:,:,grid_dim["pher"]]
    food = grid[:,:,grid_dim["food"]]
    # pick out worms that are alive and not an egg
    not_egg = np.array(np.where((df[:,p2i["alive"]]==1) & (df[:,p2i["stage"]]!=s2i["egg"])))[0]
    
    # list out food, pheromones, and genes at the location of each worm
    if len(not_egg) > 0:
        food_loc = np.ravel(food, order="F")[df[not_egg, p2i["loc"]].astype(int)]
        food_loc[food_loc > var["food_max"]] = var["food_max"]
        pher_loc = np.ravel(phers, order="F")[df[not_egg, p2i["loc"]].astype(int)]
        pher_loc[pher_loc > var["pher_max"]] = var["pher_max"]
        smell_gene = (df[not_egg,p2i["smell_1"]] + df[not_egg,p2i["smell_2"]])/2
        
        # must move if no food, chance of moving otherwise is smell*(1-food) + (1-smell)*pher
        move_conditions = np.c_[(food_loc<1),
                                smell_gene*(1-food_loc/var["food_max"]) + (1-smell_gene)*pher_loc/var["pher_max"]]
        move_prob = np.apply_along_axis(lambda row: row[0] if row[0] else row[1], 1, move_conditions)
        move_probs = np.c_[move_prob, 1-move_prob]
        decision = np.apply_along_axis(lambda row: np.random.choice([True,False], p=[row[0],row[1]]), 1, move_probs)
        df[not_egg, p2i["decision"]] = decision


# Run the Program
def run(iterations, food_start=500, food_len=10, space_between=10, patches=5, food_max=2000, seed=30,
        food_growth=[0.1,0], pher_decay=-0.5, pop_size=200, energy=5, pher_max=500, genes=10, eggs=(1/12),
        grow_time=[18,33,51,63,87,111,183,2103,3639], dauer_weight=0.5, food_eaten=[0,1,2,2,4,4,8,16,8],
        smell_weight=0.05, mutation_rate=0.001, gender=[0,1], dauer_gene=[0,35], num_patches=10,
        pher=[0,0.25,0.5,0.5,1,1,2,4,2], genders_prob=[[0.99,0.01], [0.5,0.5]], smell_gene=[0.5,0.05],
        gender_prob=0, energy_used=[0,0.5,1,1,2,0,4,8,4], food_repop=(1/15), sperm_bias=0.014, dictionary=False,
        save=[1,250,500,1000,1500,2000,5000,10000,15000,20000,25000,30000], food_freq=(math.pi/4380),
        food_amp=0, dauer_age=2880, L2d_cutoff=0.90, pop_max=1000000, dauer_die=0.98, gp_map=3, save_freq=100):
    
    if dictionary:
        all_dict = dictionary

    else:
        # check some variables first
        assert smell_gene[0] <= 1 and smell_gene[0] >= 0
        assert food_repop >= 0 and food_repop <= 1
        assert food_amp >= 0 and food_amp <= min(food_repop, 1-food_repop)
        
        # variables
        var = {"food_start":food_start, "food_len":food_len, "space_between":space_between, "patches":patches,
               "food_max": food_max, "food_growth":food_growth, "pher_decay":pher_decay, "pop_size":pop_size,
               "energy":energy, "pher_max":pher_max, "genes":genes, "eggs":eggs, "grow_time":grow_time,
               "dauer_weight":dauer_weight, "food_eaten":food_eaten, "smell_weight":smell_weight, "pher":pher,
               "mutation_rate":mutation_rate, "gender":gender, "dauer_gene":dauer_gene, "pop_max":pop_max,
               "genders_prob":genders_prob, "smell_gene":smell_gene, "gender_prob":genders_prob[gender_prob],
               "energy_used":energy_used, "food_repop":food_repop, "grid_len":(food_len + space_between)*patches,
               "num_patches":num_patches, "sperm_bias":sperm_bias, "data":0, "iter":0, "seed":seed, "save":save,
               "food_amp":food_amp, "food_freq":food_freq, "dauer_age":dauer_age, "L2d_cutoff":L2d_cutoff,
               "dauer_die":dauer_die, "gp_map":gp_map, "save_freq":save_freq}
        
        # start off the random number generator
        np.random.seed(var["seed"])
        
        # stages
        stage = ["egg", "L1", "L2", "L2d", "L3", "dauer", "L4", "adult", "old"]
        s2i = {x:i for i,x in enumerate(stage)}
        i2s = {i:x for i,x in enumerate(stage)}
        
        # genders
        g2i = {"herm":0, "female":0, "male":1}
    
        # x and y
        xx = np.tile(np.arange(var["grid_len"]), var["grid_len"])
        yy = np.repeat(np.arange(var["grid_len"]), var["grid_len"])
        
        # food
        food = np.zeros((var["grid_len"], var["grid_len"]), order = "F")
        corners = [i for i in range(0, var["grid_len"], var["food_len"] + var["space_between"])]
        food_patches = np.array([[i,j] for i in corners for j in corners])
        which_patches = np.random.choice(range(len(food_patches)), size=var["num_patches"], replace=False)
        for i in which_patches:
            x = food_patches[i][0]
            y = food_patches[i][1]
            food[x:(x + var["food_len"]), y:(y + var["food_len"])] = var["food_start"]
            
        # grid
        grid_dim = {"x":0, "y":1, "food":2, "pher":3, "f_egg":4, "f_L1":5, "f_L2":6, "f_L2d":7, "f_L3":8,
                    "f_dauer":9, "f_L4":10, "f_adult":11, "f_old":12, "m_egg":13, "m_L1":14, "m_L2":15,
                    "m_L2d":16, "m_L3":17, "m_dauer":18, "m_L4":19, "m_adult":20, "m_old":21}
        grid = np.zeros((var["grid_len"], var["grid_len"], len(grid_dim)), order = "F")
        grid[:, :, grid_dim["x"]] = xx.reshape((var["grid_len"], var["grid_len"]), order = "F")
        grid[:, :, grid_dim["y"]] = yy.reshape((var["grid_len"], var["grid_len"]), order = "F")
        grid[:, :, grid_dim["food"]] = food
        
        # neighbors
        north = np.zeros(np.shape(grid), order = "F")
        south = np.zeros(np.shape(grid), order = "F")
        west = np.zeros(np.shape(grid), order = "F")
        east = np.zeros(np.shape(grid), order = "F")
        
        # properties
        prop = ["name", "gender", "food_count", "stage", "x_loc", "y_loc", "loc", "energy", "dauer_1",
                "dauer_2", "smell_1", "smell_2", "parent_1", "parent_2", "L1", "L2", "L2d", "L3", "dauer",
                "L4", "adult", "old", "alive", "decision"]
        prop += ["gene_" + str(i) for i in range(var["genes"]*2)]
        p2i = {x:i for i,x in enumerate(prop)}
        
        # dataframe
        df=np.nan*np.zeros((var["pop_max"],len(prop)),order="F")
        df[:,p2i["name"]]=np.arange(var["pop_max"])
        df[:var["pop_size"],p2i["gender"]]=np.random.choice(var["gender"],p=var["gender_prob"],size=var["pop_size"])
        df[:,p2i["food_count"]]=0
        df[:,p2i["stage"]]=0
        worm_patch = food_patches[np.random.choice(which_patches,size=var["pop_size"])]
        df[:var["pop_size"],p2i["x_loc"]]=np.random.choice(range(var["food_len"]),size=var["pop_size"]) + worm_patch[:,0]
        df[:var["pop_size"],p2i["y_loc"]]=np.random.choice(range(var["food_len"]),size=var["pop_size"]) + worm_patch[:,1]
        df[:var["pop_size"],p2i["loc"]]=np.ravel_multi_index([df[:var["pop_size"],p2i["x_loc"]].astype(int),df[:var["pop_size"],p2i["y_loc"]].astype(int)],grid.shape[:2],order="F")
        df[:,p2i["energy"]]=var["energy"]*var["energy_used"][s2i["L1"]]
        df[:var["pop_size"],p2i["dauer_1"]]=np.random.uniform(var["dauer_gene"][0],var["dauer_gene"][1],var["pop_size"])
        df[:var["pop_size"],p2i["dauer_2"]]=np.random.uniform(var["dauer_gene"][0],var["dauer_gene"][1],var["pop_size"])
        new_smell_1 = np.random.normal(var["smell_gene"][0],var["smell_gene"][1],size=var["pop_size"])
        new_smell_1[new_smell_1 > 1] = 1
        new_smell_1[new_smell_1 < 0] = 0
        new_smell_2 = np.random.normal(var["smell_gene"][0],var["smell_gene"][1],size=var["pop_size"])
        new_smell_2[new_smell_2 > 1] = 1
        new_smell_2[new_smell_2 < 0] = 0        
        df[:var["pop_size"],p2i["smell_1"]]=new_smell_1
        df[:var["pop_size"],p2i["smell_2"]]=new_smell_2
        df[:var["pop_size"],p2i["parent_1"]]=-1
        df[:var["pop_size"],p2i["parent_2"]]=-1
        df[:,p2i["L1"]:p2i["gene_0"]]=0
        df[:var["pop_size"],p2i["alive"]]=1
        for i in range(0,var["genes"]*2,2):
            df[:var["pop_size"], p2i["gene_0"]+i] = (df[:var["pop_size"], p2i["name"]]*2-1)
        for i in range(1,var["genes"]*2,2):
            df[:var["pop_size"], p2i["gene_0"]+i] = (df[:var["pop_size"], p2i["name"]]*2)
        
        # mates array
        mates = np.array([], dtype=np.int64).reshape(0,2)
        
        # dictionary for all variables
        all_dict = {"par":var, "stage_list":stage, "s_to_i":s2i, "i_to_s":i2s, "g_to_i":g2i,
                    "grid_layer":grid_dim, "grid_amt":grid, "g_north":north, "g_south":south, "g_west":west,
                    "g_east":east, "char":prop, "p_to_i":p2i, "array":df, "mated":mates}
    
    # run the simulations
    for i in range(iterations):
        metabolism(all_dict["array"], all_dict["s_to_i"], all_dict["p_to_i"], all_dict["par"])
        deposit_pher(all_dict["grid_amt"], all_dict["par"], all_dict["s_to_i"], all_dict["grid_layer"], all_dict["p_to_i"], all_dict["array"])
        update_nbrs(all_dict["grid_amt"], all_dict["g_north"], all_dict["g_south"], all_dict["g_west"], all_dict["g_east"], all_dict["par"])
        decide(all_dict["grid_amt"], all_dict["array"], all_dict["par"], all_dict["s_to_i"], all_dict["grid_layer"], all_dict["g_north"], all_dict["g_south"], all_dict["g_west"], all_dict["g_east"], all_dict["p_to_i"])
        move(all_dict["grid_amt"], all_dict["array"], all_dict["par"], all_dict["s_to_i"], all_dict["grid_layer"], all_dict["g_north"], all_dict["g_south"], all_dict["g_west"], all_dict["g_east"], all_dict["p_to_i"])
        eat(all_dict["array"], all_dict["grid_amt"], all_dict["par"], all_dict["s_to_i"], all_dict["grid_layer"], all_dict["p_to_i"])
        grow(all_dict["array"], all_dict["par"], all_dict["s_to_i"], all_dict["p_to_i"], all_dict["grid_amt"], all_dict["grid_layer"], all_dict["g_north"], all_dict["g_south"], all_dict["g_west"], all_dict["g_east"])
        all_dict["mated"] = list_sperm(all_dict["mated"], all_dict["s_to_i"], all_dict["char"], all_dict["array"], all_dict["p_to_i"])
        all_dict["array"] = reproduce(all_dict["array"], all_dict["mated"], all_dict["par"], all_dict["s_to_i"], all_dict["g_to_i"], all_dict["char"], all_dict["p_to_i"])
        pass_time(all_dict["array"], all_dict["s_to_i"], all_dict["p_to_i"])
        grow_food(all_dict["grid_amt"], all_dict["par"], all_dict["grid_layer"])
        decay_pheromones(all_dict["grid_amt"], all_dict["par"], all_dict["grid_layer"])
        update_grid(all_dict["grid_amt"], all_dict["par"], all_dict["stage_list"], all_dict["grid_layer"], all_dict["p_to_i"], all_dict["array"])
        all_dict["par"]["iter"] += 1
        if all_dict["par"]["iter"] in all_dict["par"]["save"]:
            # take a snapshot of the current information
            file_name = "all_info_saved_iter_" + str(all_dict["par"]["iter"]) + ".p"
            new_file = open(file_name, "wb")
            pickle.dump(all_dict, new_file)
            new_file.close()
            print("Information Saved - Iteration " + str(all_dict["par"]["iter"]))
        if all_dict["par"]["iter"] % all_dict["par"]["save_freq"] == 0:
            # add onto the lineage tracking file
            file = open("lineage_tracking.txt", "a+")
            file.write(str(all_dict["par"]["iter"]) + " ")
            # find the lineages of each worm and their frequencies
            alive = np.array(np.where(all_dict["array"][:,all_dict["p_to_i"]["alive"]]==1))[0]
            lineage = copy.copy(stats.mode(all_dict["array"][alive, all_dict["p_to_i"]["gene_0"]:], axis=1)[0])
            lineage = lineage.flatten().astype(int)
            lineage[lineage%2==1] += 1
            lineage = (lineage/2).astype(int)
            counts = np.bincount(lineage)
            counts = np.round(counts/np.sum(counts), decimals=3)
            file.write(" ".join(map(str, counts)))
            file.write("\n")
            file.close()
            
            # add onto the allele tracking file
            file = open("allele_tracking.txt", "a+")
            file.write(str(all_dict["par"]["iter"]) + " ")
            # find the alleles of each worm and their frequencies
            dauer_1 = np.round(all_dict["array"][alive, all_dict["p_to_i"]["dauer_1"]])
            dauer_2 = np.round(all_dict["array"][alive, all_dict["p_to_i"]["dauer_2"]])
            dauer_genes = np.concatenate((dauer_1, dauer_2)).astype(int)
            counts = np.bincount(dauer_genes)
            counts = np.round(counts/np.sum(counts), decimals=3)
            file.write(" ".join(map(str, counts)))
            file.write("\n")
            file.close()            
        
    return(all_dict)


if __name__=="__main__":
    # the first argument is the folder location
    os.chdir(sys.argv[1])
    # the next arguments are the seed, parameter to change, and its value
    my_data = eval("run(30000, seed=" + sys.argv[2] + ", " + sys.argv[3] + "=" + sys.argv[4] + ")")
    