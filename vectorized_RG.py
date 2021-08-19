###### Vectorized Version by RG ######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pdb
import pickle
from scipy import stats
import math
import seaborn as sns
import os
import pylab


# Food Growth
def grow_food(grid, var, grid_dim):
    """ Bacterial growth occurs logistically at every time step.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters. 
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    """
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
    """ Pheromones decay exponentially at every time step.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters. 
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    """
    grid[:,:,grid_dim["pher"]] = grid[:,:,grid_dim["pher"]]*np.exp(var["pher_decay"])
    grid[:,:,grid_dim["pher"]][grid[:,:,grid_dim["pher"]]<0] = 0

# Update Worm Locations
def update_grid(grid, var, stage, grid_dim, p2i, df):
    """ Update the grid with the most recent positions of all the worms.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    stage : a list
        Lists all the stages of a worm, starting from "egg" and ending with a stage called "old."
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    """
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
    """ Finds all the adult males and adult herm/females in the same location and adds to a list of mates.
    
    Parameters
    ----------
    mates : a 2D numpy array
        Contains a list of mates as ordered pairs [herm/female, male] in each row.
    s2i : a dictionary
        Translates from worm stage to an index.
    prop : a list
        Lists all the properties of a worm (eg name, gender, etc).
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    
    Returns
    -------
    mates : a 2D numpy array
        Contains a list of mates as ordered pairs [herm/female, male] in each row.
    """
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
    """ Uses the main grid to create four nearly identical North, South, West, and East grids.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    north : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit north. 
    south : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit south.
    west : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit west.
    east : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit east.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
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
    """ Subtracts energy from all worms proportional to their stage and kills off worms that have no energy.
    
    Parameters
    ----------
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    s2i : a dictionary
        Translates from worm stage to an index. 
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
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
    """ Worms decide in which direction they will travel and then they move there.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    s2i : a dictionary
        Translates from worm stage to an index. 
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    north : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit north. 
    south : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit south.
    west : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit west.
    east : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit east.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."        
    """
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
    """ Worms will deposit pheromones proportional to their size at every time step.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    s2i : a dictionary
        Translates from worm stage to an index. 
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    """
    every = np.array(np.where((df[:,p2i["alive"]]==1) & (df[:,p2i["stage"]]!=s2i["egg"])))[0]
    # pher amount based on stage and placed in the correct location
    pher_amt = np.bincount(df[every,p2i["loc"]].astype(int), weights = np.array(var["pher"])[df[every,p2i["stage"]].astype(int)])
    new_pher = np.zeros(grid[:,:,1].shape, order = "F")
    # fit pher amounts to grid shape and add to main grid
    new_pher[np.unravel_index(np.arange(len(pher_amt)), new_pher.shape, order = "F")] = pher_amt
    grid[:,:,grid_dim["pher"]] += new_pher


# Worms Advance Stages
def grow(df, var, s2i, p2i, grid, grid_dim, north, south, west, east):
    """ Worms will transition into the next life stage given they have eaten enough food. Some must decide which stage will be next. Some will die.
    
    Parameters
    ----------
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    s2i : a dictionary
        Translates from worm stage to an index.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms. 
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    north : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit north. 
    south : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit south.
    west : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit west.
    east : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit east.        
    """
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
    """ Worms will eat food at every time step (if it is available) and portions taken will be proportional to their stage.
    
    Parameters
    ----------
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    s2i : a dictionary
        Translates from worm stage to an index.
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."        
    """
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
    """ Hermaphrodites/females will lay eggs if they have energy/sperm. Eggs will take on properties based on the parents and inherit genes with the chance for mutation.
    
    Parameters
    ----------
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    mates : a 2D numpy array
        Contains a list of mates as ordered pairs [herm/female, male] in each row.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    s2i : a dictionary
        Translates from worm stage to an index.
    g2i : a dictionary
        Translates from worm sex (gender) to an index.
    prop : a list
        Lists all the properties of a worm (eg name, gender, etc).
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    
    Returns
    -------
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    """
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
    """ Worms will count how long they've spent in each stage.
    
    Parameters
    ----------
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    s2i : a dictionary
        Translates from worm stage to an index.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."        
    """
    not_egg = np.array(np.where((df[:,p2i["alive"]]==1) & (df[:,p2i["stage"]]!=s2i["egg"])))[0]
    # convert from stage index to property index
    stages_index = df[not_egg, p2i["stage"]].astype(int)
    property_index = stages_index + p2i["L1"] - 1
    df[not_egg, property_index] += 1


# Worms Decide to Travel
def decide(grid, df, var, s2i, grid_dim, north, south, west, east, p2i):
    """ Worms will decide whether or not to travel to a new location.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    s2i : a dictionary
        Translates from worm stage to an index.
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    north : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit north. 
    south : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit south.
    west : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit west.
    east : a 3D numpy array
        Contains information identical to the "grid" but shifted one unit east.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."        
    """
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
    """ The master function that runs the program and calls all the functions above. Variables are created within this function and then passed into the other functions. 
    
    Parameters
    ----------
    iterations : a positive integer
        The number of times to loop through all the functions in the program. Represents hours. Usually set to 30,000.
    food_start : an integer or a float
        The amount of bacteria in each square at the beginning of every new patch (or when a new patch appears).
    food_len : an integer
        The length of the side of a food patch in terms of spaces on the grid. They are square.
    space_between : an integer
        The length of the space between food patches in terms of spaces on the grid.
    patches : an integer
        The number of food patches per row/column of the grid. Used to determine the grid dimensions. 
    food_max : an integer or a float
        The carrying capacity for bacteria per space on the grid and also per patch in that location.
    seed : an integer
        Controls the random choices made throughout the program. Selecting the same seed will produce the same outcome.
    food_growth : a list of two non-negative integers or floats
        The first number is the growth rate of the bacteria. The second number introduces/controls the fluctuation of that growth rate based on a normal distribution.
    pher_decay : a negative integer or float
        The rate of decay of all pheromones on the grid at each time step.
    pop_size : a positive integer
        The size of the initial population of worms, and thus, the number of unique lineages throughout the simulation.
    energy : a positive integer or float
        The factor used to multiply by the energy proportions for each life stage to determine the maximum energy a worm may have after molting.
    pher_max : an integer or float
        The maximum amount of pheromones that can be perceived by a worm.
    genes : a positive integer
        The number of neutral genes each worm stores. More genes will improve lineage tracking, but take up more storage space.
    eggs : a float between 0 and 1
        The probability an adult female/herm will lay an egg at each time step, given she has enough energy (and sperm). Re-calculated during every iteration.
    grow_time : a list of nine increasing integers or floats (each bigger than the last)
        The total amount of food that must be consumed by a worm in its lifetime before it reaches the next stage. In order of the stages.
    dauer_weight : a positive integer or float
        The size of the mutation created in the dauer gene during reproduction, assuming there is a mutation. Based on a normal distribution centered at zero.
    food_eaten : a list of nine integers or floats
        The portion of food a worm will eat (if available) at each life stage, in order of the stages.
    smell_weight : a positive integer or float
        The size of the mutation created in the smell gene during reproduction, assuming there is a mutation. Based on a normal distribution centered at zero.
    mutation_rate : a float between 0 and 1
        The probability that each dauer and smell gene of each egg individually will have a mutation.
    gender : the list [0, 1]
        The zero refers to females/hermaphrodites and the one refers to males. A worm property.
    dauer_gene : a list of two integers or floats (the smaller one first)
        The range of dauer gene values in the initial worm population. The dauer gene values will be uniformly distributed across this range to start. 
    num_patches : a positive integer
        The number of patches that a simulation will have at the beginning. Must not exceed the number of possible locations for patches.
    pher : a list of nine integers or floats
        The amount of pheromones a worm will release at every time step, based on and in order of the life stages
    genders_prob : the nested list [[0.99, 0.01], [0.5, 0.5]]
        The first interior list [0.99, 0.01] is the probability that a worm will be hermaphrodite or male, respectively. The chosen reproductive system is androdioecy.
        The second interior list [0.5, 0.5] is the probability that a worm will be female or male, respectively. The chosen reproductive system is dioecy.
    smell_gene : a list of two integers or floats
        The distribution of smell gene values in the initial worm population. Uses a normal distribution centered around the first integer/float with st dev of the second integer/float.
    gender_prob : either 0 or 1
        The choice of reproductive system. Select zero and the chosen reproductive system is androdioecy. Select one and the chosen reproductive system is dioecy.
    energy_used : a list of nine integers or floats
        The amount of energy each worm spends/metabolizes at each time step, in order of the life stages. Also helps determine the maximum energy a worm may have after molting.
    food_repop : a float between 0 and 1
        The probability that a new patch of food will appear at each time step. The midline of the patch repopulation probability / seasonality sine curve.
    sperm_bias : a float between 0 and 1
        The probability that a female/hermaphrodite with sperm will choose to use it. Maintains low male presence in hermaphrodite populations. Should be set to 1 in the male/female case.
    dictionary : either a boolean (False if no dictionary) or a dictionary
        The master dictionary that contains all variables for running an experiment, passed in from a previous simulation, with the intention of continuing that simulation. Note the seed will change undetected.
        If passing in a dictionary from a previous simulation, it must have all the same variables as listed in the experimental setup below. Then, you can "pick up where you left off."
    save : a list of positive integers
        The time points at which snapshots of information will be taken. At these points, a pickle of the master dictionary will be stored and you will be informed in the output.
    food_freq : an integer or float
        The period of the patch repopulation probability / seasonality sine curve, determines the frequency of the fluctuations in food patch replacement. The default is one year.
    food_amp : an integer or float between 0 and min(food_repop, 1 - food_repop)
        The amplitude of the patch repopulation probability / seasonality sine curve, determines the intensity of the fluctuations in food patch replacement. The default is zero or no fluctuation.
    dauer_age : an integer
        The number of time steps a worm in dauer will survive before dying of starvation. The default is 4 months.
    L2d_cutoff : a float between 0 and 1
        The threshold for determing which worms have spent too long in L2d, based on their genetics and the genotype-to-phenotype mapping component. These worms will enter dauer. See "prob_dauer" graph.
    pop_max : a large integer
        The size of the numpy array that contains all the living worms in the simulation. An array too small will create an error because it cannot contain all living worms. A very large array may be less efficient.
    dauer_die : a float between 0 and 1
        The probability that a worm entering dauer will immediately be culled.
    gp_map : an integer or float
        The genotype-to-phenotype mapping component that controls the strictness of the relationship between the dauer gene and the probability a worm actually enters dauer. See "prob_dauer" graph.
    save_freq : a positive integer
        The frequency with which lineage tracking and allele tracking is performed and stored. This information is recorded every 100 time steps by default.
    
    Returns
    -------
    all_dict : a dictionary
        The master dictionary that contains all the variables resulting at the end of the experiment including var, stage, s2i, i2s, g2i, grid_dim, grid, north, south, west, east, prop, p2i, df, and mates.
        This information is also periodically stored in a pickle, according to the paramter "save," and can be used to run continuations of the same experimental trial.
    """
    if dictionary:
        all_dict = dictionary

    else:
        # check some variables first
        assert smell_gene[0] <= 1 and smell_gene[0] >= 0
        assert food_repop >= 0 and food_repop <= 1
        assert food_amp >= 0 and food_amp <= min(food_repop, 1-food_repop)
        
        # create a variables dictionary
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
        
        # list all the life stages
        stage = ["egg", "L1", "L2", "L2d", "L3", "dauer", "L4", "adult", "old"]
        s2i = {x:i for i,x in enumerate(stage)}
        i2s = {i:x for i,x in enumerate(stage)}
        
        # list the genders
        g2i = {"herm":0, "female":0, "male":1}
    
        # x and y
        xx = np.tile(np.arange(var["grid_len"]), var["grid_len"])
        yy = np.repeat(np.arange(var["grid_len"]), var["grid_len"])
        
        # setup the food to go on the grid later
        food = np.zeros((var["grid_len"], var["grid_len"]), order = "F")
        corners = [i for i in range(0, var["grid_len"], var["food_len"] + var["space_between"])]
        food_patches = np.array([[i,j] for i in corners for j in corners])  # all possible locations
        which_patches = np.random.choice(range(len(food_patches)), size=var["num_patches"], replace=False)  # chosen starting locations
        for i in which_patches:
            # populate those chosen starting locations
            x = food_patches[i][0]
            y = food_patches[i][1]
            food[x:(x + var["food_len"]), y:(y + var["food_len"])] = var["food_start"]
            
        # create the grid with many layers
        grid_dim = {"x":0, "y":1, "food":2, "pher":3, "f_egg":4, "f_L1":5, "f_L2":6, "f_L2d":7, "f_L3":8,
                    "f_dauer":9, "f_L4":10, "f_adult":11, "f_old":12, "m_egg":13, "m_L1":14, "m_L2":15,
                    "m_L2d":16, "m_L3":17, "m_dauer":18, "m_L4":19, "m_adult":20, "m_old":21}
        grid = np.zeros((var["grid_len"], var["grid_len"], len(grid_dim)), order = "F")
        grid[:, :, grid_dim["x"]] = xx.reshape((var["grid_len"], var["grid_len"]), order = "F")
        grid[:, :, grid_dim["y"]] = yy.reshape((var["grid_len"], var["grid_len"]), order = "F")
        grid[:, :, grid_dim["food"]] = food  # add the food on there
        
        # create empty neighboring grids
        north = np.zeros(np.shape(grid), order = "F")
        south = np.zeros(np.shape(grid), order = "F")
        west = np.zeros(np.shape(grid), order = "F")
        east = np.zeros(np.shape(grid), order = "F")
        
        # list the properties of each worm
        prop = ["name", "gender", "food_count", "stage", "x_loc", "y_loc", "loc", "energy", "dauer_1",
                "dauer_2", "smell_1", "smell_2", "parent_1", "parent_2", "L1", "L2", "L2d", "L3", "dauer",
                "L4", "adult", "old", "alive", "decision"]
        prop += ["gene_" + str(i) for i in range(var["genes"]*2)]
        p2i = {x:i for i,x in enumerate(prop)}
        
        # setup the dataframe (numpy array) containing the initial worms and their information
        df=np.nan*np.zeros((var["pop_max"],len(prop)),order="F")
        df[:,p2i["name"]]=np.arange(var["pop_max"])
        df[:var["pop_size"],p2i["gender"]]=np.random.choice(var["gender"],p=var["gender_prob"],size=var["pop_size"])
        df[:,p2i["food_count"]]=0
        df[:,p2i["stage"]]=0  # always start as an egg
        worm_patch = food_patches[np.random.choice(which_patches,size=var["pop_size"])]  # each worm picks a patch
        df[:var["pop_size"],p2i["x_loc"]]=np.random.choice(range(var["food_len"]),size=var["pop_size"]) + worm_patch[:,0]
        df[:var["pop_size"],p2i["y_loc"]]=np.random.choice(range(var["food_len"]),size=var["pop_size"]) + worm_patch[:,1]
        df[:var["pop_size"],p2i["loc"]]=np.ravel_multi_index([df[:var["pop_size"],p2i["x_loc"]].astype(int),df[:var["pop_size"],p2i["y_loc"]].astype(int)],grid.shape[:2],order="F")
        df[:,p2i["energy"]]=var["energy"]*var["energy_used"][s2i["L1"]]
        df[:var["pop_size"],p2i["dauer_1"]]=np.random.uniform(var["dauer_gene"][0],var["dauer_gene"][1],var["pop_size"])
        df[:var["pop_size"],p2i["dauer_2"]]=np.random.uniform(var["dauer_gene"][0],var["dauer_gene"][1],var["pop_size"])
        new_smell_1 = np.random.normal(var["smell_gene"][0],var["smell_gene"][1],size=var["pop_size"])
        new_smell_1[new_smell_1 > 1] = 1  # to ensure this gene value stays between 0 and 1
        new_smell_1[new_smell_1 < 0] = 0  # to ensure this gene value stays between 0 and 1
        new_smell_2 = np.random.normal(var["smell_gene"][0],var["smell_gene"][1],size=var["pop_size"])
        new_smell_2[new_smell_2 > 1] = 1  # to ensure this gene value stays between 0 and 1
        new_smell_2[new_smell_2 < 0] = 0  # to ensure this gene value stays between 0 and 1    
        df[:var["pop_size"],p2i["smell_1"]]=new_smell_1
        df[:var["pop_size"],p2i["smell_2"]]=new_smell_2
        df[:var["pop_size"],p2i["parent_1"]]=-1  # the original worms don't have a known parent
        df[:var["pop_size"],p2i["parent_2"]]=-1  # the original worms don't have a known parent
        df[:,p2i["L1"]:p2i["gene_0"]]=0
        df[:var["pop_size"],p2i["alive"]]=1
        for i in range(0,var["genes"]*2,2):
            df[:var["pop_size"], p2i["gene_0"]+i] = (df[:var["pop_size"], p2i["name"]]*2-1)
        for i in range(1,var["genes"]*2,2):
            df[:var["pop_size"], p2i["gene_0"]+i] = (df[:var["pop_size"], p2i["name"]]*2)
        
        # create an empty array for the mates list
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
        all_dict["par"]["iter"] += 1  # keep track of the iteration number
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
            dauer_genes[dauer_genes < 0] = 0
            counts = np.bincount(dauer_genes)
            file.write(" ".join(map(str, counts)))
            file.write("\n")
            file.close()            
        
    return(all_dict)


# Graph the Food Locations
def food_map(grid, var, grid_dim):
    """ Create a 2D heatmap of the torus, showing food density by color intensity.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    grid_dim : a dictionary
        Enumerates all layers of the grid. 
    """
    x, y = np.mgrid[slice(0,var["grid_len"]), slice(0,var["grid_len"])]
    z = copy.copy(grid[:,:,grid_dim["food"]])
    z[z==0] = np.nan
    plt.pcolormesh(x, y, z, cmap="Blues", shading="auto", vmin=0)
    plt.title("Food Density")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis([0,var["grid_len"],0,var["grid_len"]])
    plt.colorbar(label="Units of Bacteria")

# Graph the Pheromone Locations
def pher_map(grid, var, grid_dim):
    """ Create a 2D heatmap of the torus, showing pheromone density by color intensity.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    grid_dim : a dictionary
        Enumerates all layers of the grid. 
    """
    x, y = np.mgrid[slice(0,var["grid_len"]), slice(0,var["grid_len"])]
    z = copy.copy(grid[:,:,grid_dim["pher"]])
    z[z==0] = np.nan
    plt.pcolormesh(x, y, z, cmap="Blues", shading="auto", vmin=0)
    plt.title("Pheromone Density")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis([0,var["grid_len"],0,var["grid_len"]])
    plt.colorbar()

# Graph the Probability of Traveling
def prob_move(var):
    """ Create a heatmap showing the probability of a worm to travel based on the amount of food and pheromones near it.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters. 
    """
    # assumptions: the amount of food/pheromone has a set max and the average smell gene is constant
    x, y = np.mgrid[slice(0,(var["food_max"]+1)), slice(0,(var["pher_max"]+1))]
    z = np.zeros(((var["food_max"]+1),(var["pher_max"]+1)))
    z[0,:] = 1
    for i in range(1,(var["food_max"]+1)):
        for j in range(0,(var["pher_max"]+1)):
            z[i,j] = var["smell_gene"][0]*(1-(i/var["food_max"])) + (1-var["smell_gene"][0])*(j/var["pher_max"])
    plt.pcolormesh(x, y, z, cmap="Blues", shading="auto")
    plt.title("Probability of Traveling")
    plt.xlabel("Amount of Food")
    plt.ylabel("Amount of Pheromones")
    plt.axis([0,var["food_max"],0,var["pher_max"]])
    plt.colorbar(ticks=np.arange(0,1.1,0.1))

# Graph the Probability of Going into Dauer
def prob_dauer(var, time_spent):
    """ Create a heatmap showing the probability of a worm going into dauer based on possible gene values and the amount of pheromones near it. The vertical line shows the average amount of time spent in L2d.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    time_spent : an integer or float
        The average amount of time that all worms throughout a particular simulation spend in L2d.
    """
    x, y = np.mgrid[slice(var["dauer_gene"][0],var["dauer_gene"][1]+0.25,0.25), slice(0,var["pher_max"]+1)]
    x_num = (var["dauer_gene"][1]-var["dauer_gene"][0])*4+1
    z = np.zeros((x_num,var["pher_max"]+1))
    for i,k in zip(np.arange(var["dauer_gene"][0],var["dauer_gene"][1]+0.25,0.25),range(x_num)):
        for j in range(var["pher_max"]+1):
            prob = 1/(1 + np.exp((i - time_spent)/var["gp_map"]))
            z[k,j] = 0.5*prob + 0.5*j/var["pher_max"]
    z[z >= var["L2d_cutoff"]] = 1
    plt.pcolormesh(x, y, z, cmap="coolwarm", shading="auto")
    plt.axvline(x=time_spent, color="black")
    plt.text(time_spent-1.5,var["pher_max"]/2-50,"time spent",rotation=90)
    plt.text(var["dauer_gene"][0]+2,var["pher_max"]-50,"cutoff")
    plt.text(var["dauer_gene"][1]-12,var["pher_max"]-50,"map value = "+str(var["gp_map"]))
    plt.title("Probability of Going into Dauer")
    plt.xlabel("Initial Dauer Gene Values")
    plt.ylabel("Amount of Pheromones")
    plt.axis([var["dauer_gene"][0],var["dauer_gene"][1],0,var["pher_max"]])
    plt.colorbar(ticks=np.arange(0,1.1,0.1))

# How Many Worms per Gender
def num_gender(var, g2i, p2i, df):
    """ Create a pie chart showing the number of males and females/herms alive at any given time point.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    g2i : a dictionary
        Translates from worm sex (gender) to an index.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    """
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    num_gen = [np.sum(df[alive,p2i["gender"]]==g2i["male"]), np.sum(df[alive,p2i["gender"]]==g2i["herm"])]
    if (var["gender_prob"] == var["genders_prob"][0]):
        labels = ["male - " + str(num_gen[0]), "herm - " + str(num_gen[1])]
    else:
        labels = ["male - " + str(num_gen[0]), "female - " + str(num_gen[1])]
    patches, texts, percent = plt.pie(num_gen, autopct="%1.1f%%")
    plt.legend(patches, labels, loc="best")
    plt.axis("equal")
    plt.title("Number of Worms by Gender")
    
# How Many Worms per Stage
def num_stage(i2s, p2i, df):
    """ Create a bar chart showing the number of worms in each life stage alive at any given time point.
    
    Parameters
    ----------
    i2s : a dictionary
        Translates from an index to worm stage.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    """
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    num_stage = [np.sum(df[alive,p2i["stage"]]==stage) for stage in i2s]
    plt.bar([*range(0,len(i2s)*2,2)], num_stage, width=1, tick_label=[i2s[i] for i in range(len(i2s))])
    plt.ylabel("Number of Worms")

# Fraction of Worms that Die in Each Stage
def frac_dead(i2s, p2i, df, var):
    """ Print the fraction of worms that died in each life stage (versus the total that made it to that stage) and create a pie chart showing the number of worms that died by stage.
    
    Parameters
    ----------
    i2s : a dictionary
        Translates from an index to worm stage.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
    # collect all worms from old files
    combine = np.array([], dtype=np.int64).reshape(0,9)
    for i in range(var["data"]):
        file_name = "dead_worm_data_dump_" + str(i) + ".p"
        old_file = open(file_name, "rb")
        old_data = pickle.load(old_file)
        old_file.close()
        old_data = old_data[:,[p2i["alive"],p2i["stage"],p2i["L1"],p2i["L2"],p2i["L2d"],p2i["L3"],p2i["dauer"],p2i["L4"],p2i["adult"]]]
        combine = np.concatenate((combine, old_data))
    
    # add the rest of the worms
    subset = df[~np.isnan(df[:,p2i["gender"]]),:]
    subset = subset[:,[p2i["alive"],p2i["stage"],p2i["L1"],p2i["L2"],p2i["L2d"],p2i["L3"],p2i["dauer"],p2i["L4"],p2i["adult"]]]
    dead = subset[subset[:,0]==0,:]
    dead = np.concatenate((combine, dead))
    total = np.concatenate((combine, subset))
    
    # count how many are dead per stage
    stage, counts = np.unique(dead[:,1],return_counts=True)
    num_dead = np.zeros(9).astype(int)
    for i in range(9):
        if i in stage:
            num_dead[i] = counts[np.where(stage==i)]
    # figure out fraction that died vs total that made it
    fraction = [num_dead[i-1]/np.sum(total[:, i]>0) if np.sum(total[:, i]>0) > 0 else np.nan for i in range(2,9)]
    
    # print them all out
    print("chance of dying in each stage")
    for i in range(1,8):
        print(i2s[i] + " : " + str(fraction[i-1]))
    
    # make a pie chart of how many died in each stage
    if not np.all(np.isnan(np.array(fraction))):
        labels = [(i2s[i] + " - " + str(num_dead[i])) for i in range(len(num_dead))]
        patches, texts = plt.pie(num_dead)
        plt.legend(patches, labels, loc="best", title="Stage - Worms")
        plt.axis("equal")
        plt.title("Stages in Which Worms Died")
    
# Graph the Worm Locations
def worm_map(grid, var, grid_dim, map_type="worm"):
    """ Create a 2D heatmap of the torus, showing worm density (of a specific stage or gender) by color intensity.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    map_type : a string (choices are "worm", "male", "female", or one of the nine life stages)
        Specifies which worms will be displayed in the heatmap. The default shows all worms.
    """
    x, y = np.mgrid[slice(0,var["grid_len"]), slice(0,var["grid_len"])]
    if map_type == "worm":
        z = copy.copy(np.sum(grid[:,:,grid_dim["f_egg"]:],axis=2))
    elif map_type == "male":
        z = copy.copy(np.sum(grid[:,:,[grid_dim[i] for i in grid_dim if "m_" in i]],axis=2))
    elif map_type == "female":
        z = copy.copy(np.sum(grid[:,:,[grid_dim[i] for i in grid_dim if "f_" in i]],axis=2))
    else:
        z = copy.copy(np.sum(grid[:,:,[grid_dim[i] for i in grid_dim if map_type in i]],axis=2))
    z[z==0] = np.nan
    plt.pcolormesh(x, y, z, cmap="Blues", shading="auto", vmin=0)
    plt.title(map_type.capitalize() + " Density")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis([0,var["grid_len"],0,var["grid_len"]])
    plt.colorbar()

# Graph the Males and Females by Color
def gender_map(grid, var, grid_dim):
    """ Create a 2D heatmap of the torus, showing worm density by color intensity. Males are shown in red on top of females/herms shown in blue.
    
    Parameters
    ----------
    grid : a 3D numpy array
        Contains information about each location on the grid including position, food, pheromones, and worms.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    grid_dim : a dictionary
        Enumerates all layers of the grid.
    """
    # male data is translucent and on top, so covers over female data a bit in those locations
    x, y = np.mgrid[slice(0,var["grid_len"]), slice(0,var["grid_len"])]
    z1 = copy.copy(np.sum(grid[:,:,[grid_dim[i] for i in grid_dim if "m_" in i]],axis=2))
    z1[z1==0] = np.nan
    z2 = copy.copy(np.sum(grid[:,:,[grid_dim[i] for i in grid_dim if "f_" in i]],axis=2))
    z2[z2==0] = np.nan
    plt.figure()
    plt.subplot()
    females = plt.pcolormesh(x, y, z2, cmap="Blues", shading="auto", vmin=0)
    males = plt.pcolormesh(x, y, z1, cmap="Reds", alpha=0.5, shading="auto", vmin=0)
    plt.axis([0,var["grid_len"],0,var["grid_len"]])
    plt.colorbar(males, label="Males")
    plt.colorbar(females, label="Hermaphrodites / Females")
    plt.title("Worm Density by Gender")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

# Graph the Population Size Over Time
def worms_alive(save, all_my_data):
    """ Create a line graph showing the number of worms alive over time throughout a simulation.
    
    Parameters
    ----------
    save : a list of positive integers
        The time points at which snapshots of information were taken. Can be gathered by observing the file names.
    all_my_data : a list of master dictionaries 
        Creates a list to read in and store the master dictionary from each saved time point in a particular simulation. See "all_dict" in the function "run" above.
    """
    # create an empty list
    alive_num = []
    
    # loop through all the time points and count the number of living worms
    for i in range(len(save)):
        df = all_my_data[i]["array"]
        p2i = all_my_data[i]["p_to_i"]
        alive_num.append(np.sum(df[:,p2i["alive"]]==1))
    
    plt.plot(save, alive_num, "-o")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Number of Worms Alive")

# Calculate the Average Time Spent and Show a Boxplot
def stage_time(stage, p2i, df, var, exclude_dauer=False):
    """ Create boxplots showing the amount of time worms spend in each life stage, with averages printed on top.
    
    Parameters
    ----------
    stage : a list
        Lists all the stages of a worm, starting from "egg" and ending with a stage called "old."
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    exclude_dauer : a boolean (True or False)
        If True, excludes dauer from the boxplots, allowing the other stages to be viewed and compared more easily. 
        If False, shows the boxplot for dauer, which will be much larger than the others.
    
    Returns
    -------
    averages[2] : a float
        The average amount of time that all worms throughout a particular simulation spend in L2d.
    """
    # gather all data on stages
    time_spent = np.array([], dtype=np.int64).reshape(0,8)
    for i in range(var["data"]):
        file_name = "dead_worm_data_dump_" + str(i) + ".p"
        old_file = open(file_name, "rb")
        old_data = pickle.load(old_file)
        old_file.close()
        old_data = copy.copy(old_data[:,p2i["L1"]:p2i["alive"]])
        time_spent = np.concatenate((time_spent, old_data))
    time_spent = np.concatenate((time_spent, copy.copy(df[:,p2i["L1"]:p2i["alive"]])))
    
    # figure out the amount of time spent in each stage
    time_per_stage = [[] for i in range(8)]
    for i in range(8):
        time_per_stage[i] = time_spent[:,i][time_spent[:,i] != 0]
    time_spent[time_spent == 0] = np.nan
    if not np.all(np.isnan(time_spent)):
        averages = np.round(np.nanmean(time_spent, axis=0), decimals=3)
        if exclude_dauer:
            stages = stage[1:5] + stage[6:]
            new_time = time_per_stage[:4] + time_per_stage[5:]
            new_avg = list(averages[:4]) + list(averages[5:])
            position = [0, 50, 50, 50, 50, 50, 200, 200]
            plt.boxplot(new_time, showmeans=True)
            plt.xticks([*range(1,8)], stages)
            plt.title("Distribution of Time Spent Per Stage")
            plt.ylabel("Time Spent (hrs)")
            for i,j in zip(new_avg, range(1,8)):
                plt.text(j, i+position[j], str(np.round(i,1)), ha="center")
        else:
            position = [0, 200, 200, 200, 200, -300, 200, 300, 400]
            plt.boxplot(time_per_stage, showmeans=True)
            plt.xticks([*range(1,9)], (stage[1:]))
            plt.title("Distribution of Time Spent Per Stage")
            plt.ylabel("Time Spent (hrs)")
            for i,j in zip(averages, range(1,9)):
                plt.text(j, i+position[j], str(np.round(i,1)), ha="center")
    
    # return the L2d average value
    return(averages[2])

# Statistics for the Dauer Gene
def stats_d(p2i, df, var, time_spent):
    """ Create bar graphs showing the distribution of dauer genetics at any particular time point in a simulation. The vertical line shows the average amount of time spent in L2d.
    
    Parameters
    ----------
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    time_spent : an integer or float
        The average amount of time that all worms throughout a particular simulation spend in L2d.
    """
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    dauer_both = np.hstack((df[alive,p2i["dauer_1"]], df[alive,p2i["dauer_2"]]))
    dauer_avg = (df[alive,p2i["dauer_1"]] + df[alive,p2i["dauer_2"]])/2
    sort_dauer_both = np.unique(np.round(dauer_both), return_counts=True)
    sort_dauer_avg = np.unique(np.round(dauer_avg), return_counts=True)
    plt.bar(sort_dauer_both[0], sort_dauer_both[1]/(2*len(alive)), color="blue", label="Allele Pool", alpha=0.5)
    plt.bar(sort_dauer_avg[0], sort_dauer_avg[1]/len(alive), color="red", label="Gene Expressed", alpha=0.5)
    plt.legend()
    plt.xticks([var["dauer_gene"][0]+3,var["dauer_gene"][1]-2], ["more likely " + str(var["dauer_gene"][0]+3), "less likely " + str(var["dauer_gene"][1]-2)])
    plt.xlim(var["dauer_gene"][0],var["dauer_gene"][1]+1)
    plt.ylim(0,1)
    plt.axvline(x=time_spent, color="black")
    plt.title("L2d and Dauer Decision")
    plt.xlabel("Likelihood of L2d/Dauer")
    plt.ylabel("Fraction of Population")

# Statistics for the Smell Gene
def stats_s(p2i, df):
    """ Create bar graphs showing the distribution of travel direction genetics at any particular time point in a simulation.
    
    Parameters
    ----------
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    """
    # x axis limits include smell genes from 0.25 - 0.75
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    smell_both = np.hstack((df[alive,p2i["smell_1"]], df[alive,p2i["smell_2"]]))
    smell_avg = (df[alive,p2i["smell_1"]] + df[alive,p2i["smell_2"]])/2
    sort_smell_both = np.unique(np.round(smell_both,2), return_counts=True)
    sort_smell_avg = np.unique(np.round(smell_avg,2), return_counts=True)
    plt.bar(sort_smell_both[0], sort_smell_both[1]/(2*len(alive)), color="blue", label="Allele Pool", alpha=0.5, width=0.006)
    plt.bar(sort_smell_avg[0], sort_smell_avg[1]/len(alive), color="red", label="Gene Expressed", alpha=0.5, width=0.006)
    plt.legend()
    plt.xticks([0.28,0.5,0.72], ["neighbors", "equal", "food"])
    plt.xlim(0.25,0.75)
    plt.ylim(0,1)
    plt.title("Travel Direction Decision")
    plt.xlabel("Weight of Factors")
    plt.ylabel("Fraction of Population")

# Statistics for the Neutral Genes
def stats_g(var, p2i, df):
    """ Create bar graphs showing the number of descendants of each genetic lineage in the population at any particular time point in a simulation.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    """
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    genes = copy.copy(stats.mode(df[alive, p2i["gene_0"]:], axis=1)[0])
    genes = genes.flatten().astype(int)
    genes[genes%2==1] += 1
    genes = genes/2
    total = np.bincount(genes.astype(int))
    total = np.concatenate((total, np.zeros(var["pop_size"] - len(total)))).astype(int)
    plt.bar(np.arange(var["pop_size"]), total, width=1)
    plt.title("Genetic Line Tracking")
    plt.xlabel("Original Worm Number")
    plt.ylabel("Number of Descendants")

# Retrieve Info from Binary Format
def open_pickle(iteration):
    """ Reads in the data stored (the master dictionary) from any particular time point in a simulation. The location of this data must be specified in advance.
    
    Parameters
    ----------
    iteration : an integer
        Specifies the time point from which information will be retrieved. There must be a file saved with an information snapshot from that specific time point.
    
    Returns
    -------
    all_data : a dictionary
        The master dictionary that contains all the variables resulting at the end of the experiment including var, stage, s2i, i2s, g2i, grid_dim, grid, north, south, west, east, prop, p2i, df, and mates.
    """
    file_name = "all_info_saved_iter_" + str(iteration) + ".p"
    old_file = open(file_name, "rb")
    all_data = pickle.load(old_file)
    old_file.close()
    return(all_data)

# Fraction of Worms in Dauer by Genotype
def frac_dauer(p2i, df, var):
    """ Plot points on a graph showing the fraction of worms that went into dauer by their expressed dauer gene value. Number of worms that went into dauer versus the total number of worms in L3 or dauer.
    
    Parameters
    ----------
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
    # collect all worms from old files
    combine = np.array([], dtype=np.int64).reshape(0,4)
    for i in range(var["data"]):
        file_name = "dead_worm_data_dump_" + str(i) + ".p"
        old_file = open(file_name, "rb")
        old_data = pickle.load(old_file)
        old_file.close()
        old_data = old_data[:,[p2i["L3"],p2i["dauer"],p2i["dauer_1"],p2i["dauer_2"]]]
        combine = np.concatenate((combine, old_data))    
    combine = np.concatenate((combine, df[:,[p2i["L3"],p2i["dauer"],p2i["dauer_1"],p2i["dauer_2"]]]))
    
    # which worms went into dauer vs. L3
    dauer = np.array(np.where(combine[:, 1]>0))[0]
    L3 = np.array(np.where(combine[:, 0]>0))[0]
    
    if len(dauer)>0 or len(L3)>0:
        # find the expressed dauer gene values
        dauer_gene = np.round((combine[dauer,2] + combine[dauer,3])/2).astype(int)
        L3_gene = np.round((combine[L3,2] + combine[L3,3])/2).astype(int)
        
        for i in range(min(min(dauer_gene),min(L3_gene)), (max(max(dauer_gene),max(L3_gene))+1)):
            # what fraction of worms went into dauer
            fraction_dauer = np.sum(dauer_gene == i) / (np.sum(dauer_gene == i) + np.sum(L3_gene == i))
            plt.plot(i, fraction_dauer, marker="o", color="tab:blue")
        
        plt.xlabel("Expressed Dauer Gene Values")
        plt.ylabel("Fraction of Worms that Went into Dauer")

# Fraction of Worms in Dauer by Genotype vs. Hours Spent
def frac_dauer_map(p2i, df, var):
    """ Create a heatmap of the fraction of worms that went into dauer based on their expressed dauer gene values and their numbers of time steps spent in L2d. Only includes worms that chose L2d.
    
    Parameters
    ----------
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
    # collect all worms from old files
    combine = np.array([], dtype=np.int64).reshape(0,5)
    for i in range(var["data"]):
        file_name = "dead_worm_data_dump_" + str(i) + ".p"
        old_file = open(file_name, "rb")
        old_data = pickle.load(old_file)
        old_file.close()
        old_data = old_data[:,[p2i["L2d"],p2i["L3"],p2i["dauer"],p2i["dauer_1"],p2i["dauer_2"]]]
        combine = np.concatenate((combine, old_data))    
    combine = np.concatenate((combine, df[:,[p2i["L2d"],p2i["L3"],p2i["dauer"],p2i["dauer_1"],p2i["dauer_2"]]]))
    
    # which worms went from L2d into dauer or L3
    combine = combine[combine[:,0]>0]
    combine = np.concatenate((combine[combine[:,1]>0], combine[combine[:,2]>0]))
    
    # min and max of genes and hours spent
    genes = ((combine[:,3] + combine[:,4])/2).astype(int)
    min_gene = np.min(genes)
    max_gene = np.max(genes) + 1
    min_hours = np.min(combine[:,0]).astype(int)
    max_hours = np.max(combine[:,0]).astype(int) + 1
    
    # set up the heatmap
    x, y = np.mgrid[slice(min_gene,max_gene), slice(min_hours,max_hours)]
    z = np.zeros((max_gene-min_gene,max_hours-min_hours))

    for i in range(min_gene, max_gene):
        for j in range(min_hours, max_hours):
            # find worms with the correct gene and hours spent
            temp_combine = combine[genes==i]
            temp_combine = temp_combine[temp_combine[:,0]==j]
            if len(temp_combine) > 0:
                # calculate the fraction in dauer
                L3 = np.sum(temp_combine[:,1]>0)
                dauer = np.sum(temp_combine[:,2]>0)
                z[i-min_gene,j-min_hours] = dauer / (dauer + L3)
            else:
                z[i-min_gene,j-min_hours] = np.nan

    plt.pcolormesh(x, y, z, cmap="coolwarm", shading="auto")
    plt.title("Fraction of Worms that Went into Dauer")
    plt.xlabel("Expressed Dauer Gene Values")
    plt.ylabel("Hours Spent in L2d Before Deciding")
    plt.axis([min_gene,max_gene-1,min_hours,max_hours-1])
    plt.colorbar(ticks=np.arange(0,1.1,0.1))

# Graph the Winning Genetic Lines in Each Square
def genetic_line_map(var, df, p2i):
    """ Create a heatmap of the most common or "winning" genetic lineages in each square of the grid individually, displayed by color for the different lineages.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    """
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    x, y = np.mgrid[slice(0,var["grid_len"]), slice(0,var["grid_len"])]
    z = np.zeros((var["grid_len"],var["grid_len"]))*np.nan
    for i in range(var["grid_len"]):
        for j in range(var["grid_len"]):
            worms = alive[((df[alive,p2i["x_loc"]]==i) & (df[alive,p2i["y_loc"]]==j))]
            if len(worms) > 0:
                lineage = copy.copy(stats.mode(df[worms, p2i["gene_0"]:], axis=1)[0])
                lineage = lineage.flatten().astype(int)
                lineage[lineage%2==1] += 1
                lineage = lineage/2
                z[i,j] = stats.mode(lineage)[0]
    plt.pcolormesh(x, y, z, cmap="tab20", shading="auto", vmin=0, vmax=var["pop_size"])
    plt.axis([0,var["grid_len"],0,var["grid_len"]])
    plt.title("Winning Worm Genetic Lines")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.colorbar(label="Original Worm Number", ticks=np.arange(0,var["pop_size"]+1,var["pop_size"]/10))

# Count Dauer Worms by Genetic Line
def dauer_line(df, p2i, s2i, var):
    """ Plot points on a graph that indicate the fraction of worms in dauer for each genetic lineage, only including those that make up more than 1% of the population.
    
    Parameters
    ----------
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    s2i : a dictionary
        Translates from worm stage to an index.
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    genes = copy.copy(stats.mode(df[alive, p2i["gene_0"]:], axis=1)[0])
    genes = genes.flatten().astype(int)
    genes[genes%2==1] += 1
    genes = genes/2
    for i in range(var["pop_size"]):
        indices = alive[genes==i]
        if len(indices) > np.max([len(alive)*0.01,2]):
            frac = np.sum(df[indices,p2i["stage"]]==s2i["dauer"])/len(indices)
            plt.plot(i, frac, marker="o")
            plt.text(i, frac, str(i))
    plt.xlim(0,var["pop_size"])
    plt.title("Genetic lines with more than " + str(int(np.max([len(alive)*0.01,2]))) + " living individuals")
    plt.xlabel("Original Worm Number")
    plt.ylabel("Fraction of Worm Descendants in Dauer")

# Diversity Values of Each Square
def diversity(var, df, p2i):
    """ Create a heatmap showing the diversity of genetic lineages, in terms of Shannon index, calculated for each location on the grid. Shannon index values are limited from 0 to 4.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    """
    # the limit of Shannon index values is from 0 to 4
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    x, y = np.mgrid[slice(0,var["grid_len"]), slice(0,var["grid_len"])]
    z = np.zeros((var["grid_len"],var["grid_len"]))*np.nan
    for i in range(var["grid_len"]):
        for j in range(var["grid_len"]):
            worms = alive[((df[alive,p2i["x_loc"]]==i) & (df[alive,p2i["y_loc"]]==j))]
            if len(worms) > 0:
                lineage = copy.copy(stats.mode(df[worms, p2i["gene_0"]:], axis=1)[0])
                lineage = lineage.flatten().astype(int)
                lineage[lineage%2==1] += 1
                lines, nums = np.unique(lineage, return_counts=True)
                nums = nums/np.sum(nums)
                z[i,j] = -1*np.sum(nums*np.log(nums))
    plt.pcolormesh(x, y, z, cmap="Blues", shading="auto", vmin=0, vmax=4)
    plt.axis([0,var["grid_len"],0,var["grid_len"]])
    plt.title("Worm Diversity")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.colorbar(label="Shannon index", ticks=np.arange(0,4.5,0.5))

# Generation Number
def which_gen(df, p2i, var):
    """ Prints out the generation number of the "youngest" worm alive, considering the starting population to be the first generation. Generations are counted by repeatedly collecting groupings of parents. 
    
    Parameters
    ----------
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
    # gather data on alive, parent 1, and parent 2
    combine = np.array([], dtype=np.int64).reshape(0,4)
    for i in range(var["data"]):
        file_name = "dead_worm_data_dump_" + str(i) + ".p"
        old_file = open(file_name, "rb")
        old_data = pickle.load(old_file)
        old_file.close()
        old_data = old_data[:,[p2i["name"],p2i["alive"],p2i["parent_1"],p2i["parent_2"]]]
        combine = np.concatenate((combine, old_data))
    combine = np.concatenate((combine, df[:,[p2i["name"],p2i["alive"],p2i["parent_1"],p2i["parent_2"]]]))    
    
    # count the generations by gathering parents repeatedly
    worms = np.array(np.where(combine[:,1]==1))[0]
    count = 0
    while len(worms) > 0:
        worms = np.concatenate((combine[worms,2].astype(int), combine[worms,3].astype(int)))
        worms = np.unique(worms)
        worms = worms[worms>-1]
        worms = np.array(np.where(np.isin(combine[:,0], worms)))[0]
        count += 1
    print("This is generation number " + str(count))

# Measure of Dispersal
def clump(var, df, p2i):
    """ Create a histogram showing the dispersal distribution of genetic lineages that make up more than 1% of the population. Compared to the dispersal of a random sample of the population (vertical line).
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    df : a 2D numpy array
        Contains all worms (up to the set max) and their many properties (eg name, gender, etc).
    p2i : a dictionary
        Translates from worm property (eg name, gender, etc) to an index in the worm array called "df."
    """
    # figure out who the genetic ancestor is
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    genes = copy.copy(stats.mode(df[alive, p2i["gene_0"]:], axis=1)[0])
    genes = genes.flatten().astype(int)
    genes[genes%2==1] += 1
    genes = (genes/2).astype(int)
    
    # find clumpiness of each lineage only for lines with more than 1% of worms
    lineage = []
    for i in np.unique(genes):
        worms = alive[genes==i]
        if len(worms) > np.max([len(alive)*0.01,2]):
            worms = np.random.choice(worms, size=min(len(worms), 1000), replace=False)
            pairs = np.array([[x,y] for i,x in enumerate(worms) for j,y in enumerate(worms) if i < j])
            x_diff = np.absolute(df[pairs[:,0],p2i["x_loc"]] - df[pairs[:,1],p2i["x_loc"]])
            y_diff = np.absolute(df[pairs[:,0],p2i["y_loc"]] - df[pairs[:,1],p2i["y_loc"]])
            x_val = np.minimum(x_diff,(var["grid_len"]-x_diff))**2
            y_val = np.minimum(y_diff,(var["grid_len"]-y_diff))**2
            lineage.append(np.mean(np.sqrt(x_val + y_val)))
    
    # find the clumpiness of a random sample of the population
    sample = np.random.choice(alive, size=min(len(alive), 1000), replace=False)
    pairs = np.array([[x,y] for i,x in enumerate(sample) for j,y in enumerate(sample) if i < j])
    x_diff = np.absolute(df[pairs[:,0],p2i["x_loc"]] - df[pairs[:,1],p2i["x_loc"]])
    y_diff = np.absolute(df[pairs[:,0],p2i["y_loc"]] - df[pairs[:,1],p2i["y_loc"]])
    x_val = np.minimum(x_diff,(var["grid_len"]-x_diff))**2
    y_val = np.minimum(y_diff,(var["grid_len"]-y_diff))**2
    baseline = np.mean(np.sqrt(x_val + y_val))
    if len(lineage) > 1:
        f = sns.displot(lineage, kde=True)
    else:
        f = sns.displot(lineage, kde=False)
    plt.axvline(x=baseline, color="gray")
    f.set_axis_labels("Average Distance Between Worms", "Number of Lineages")
    plt.title("Dispersal of Genetic Lines")
    
# Rate of Food Patch Repopulation
def patch_repop(var):
    """ Plot a line graph of the probability of food patch repopulation over the time course of a simulation.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
    times = [i for i in range(0,var["iter"]+1,100)]
    y=[]
    for i in range(len(times)):
        repop_rate = var["food_amp"] * math.sin(var["food_freq"] * times[i]) + var["food_repop"]
        y.append(repop_rate)
    plt.plot(times, y)
    plt.axhline(y=var["food_repop"], color="gray")
    plt.ylim(0,1)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Chance of Patch Repopulation")

# Average Value of Dauer Gene Over Time
def dauer_over_time(var, time_spent, save, all_my_data):
    """ Create a line graph showing the change over time in the average dauer gene for the population (red) and the average dauer gene for the most common or "winning" lineage (blue) from the last time point.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    time_spent : an integer or float
        The average amount of time that all worms throughout a particular simulation spend in L2d.
    save : a list of positive integers
        The time points at which snapshots of information were taken. Can be gathered by observing the file names.
    all_my_data : a list of master dictionaries 
        Creates a list to read in and store the master dictionary from each saved time point in a particular simulation. See "all_dict" in the function "run" above.
    """
    # define some variables from the last time point
    df = all_my_data[-1]["array"]
    p2i = all_my_data[-1]["p_to_i"]
    
    # determine which lineage is the most common
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    genes = copy.copy(stats.mode(df[alive, p2i["gene_0"]:], axis=1)[0])
    genes = genes.flatten().astype(int)
    genes[genes%2==1] += 1
    genes = genes/2
    which = stats.mode(genes)[0][0]
    
    # make all the empty lists
    dauer_value = []
    dauer_value_std = []
    dauer_avg_value = []
    dauer_avg_value_std = []
    
    # loop through every time point
    for i in range(len(save)):
        # define some variables for the current time point
        df = all_my_data[i]["array"]
        p2i = all_my_data[i]["p_to_i"]
        
        # get the list of lineages and pick the worms that match the winning lineage
        alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
        genes = copy.copy(stats.mode(df[alive, p2i["gene_0"]:], axis=1)[0])
        genes = genes.flatten().astype(int)
        genes[genes%2==1] += 1
        genes = genes/2
        these = alive[genes==which]
        
        # find their average dauer gene value and standard deviation
        dauer_value.append(np.mean((df[these,p2i["dauer_1"]] + df[these,p2i["dauer_2"]])/2))
        dauer_value_std.append(np.std((df[these,p2i["dauer_1"]] + df[these,p2i["dauer_2"]])/2))
        
        # find the average dauer gene and standard deviation for the entire population
        dauer_avg_value.append(np.mean((df[alive,p2i["dauer_1"]] + df[alive,p2i["dauer_2"]])/2))
        dauer_avg_value_std.append(np.std((df[alive,p2i["dauer_1"]] + df[alive,p2i["dauer_2"]])/2))
    
    plt.errorbar(save, dauer_value, yerr=dauer_value_std, marker="o", color="b", label="Winning Lineage " + str(int(which)))
    plt.errorbar(save, dauer_avg_value, yerr=dauer_avg_value_std, marker="o", color="r", label="Population Average")
    plt.text(save[-1], dauer_value[-1]-2, str(np.round(dauer_value[-1],2)), ha="center", color="b")
    plt.text(save[-1], dauer_avg_value[-1]+1.5, str(np.round(dauer_avg_value[-1],2)), ha="center", color="r")
    plt.ylim(var["dauer_gene"][0],var["dauer_gene"][1]+1)
    plt.axhline(y=time_spent, color="black")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Average Value of Dauer Gene")
    plt.legend()

# Average Value of Smell Gene Over Time
def smell_over_time(save, all_my_data):
    """ Create a line graph showing the change over time in the average travel gene for the population (red) and the average travel gene for the most common / "winning" lineage (blue) at the last time point.
    Note that the limits on the y axis only include travel direction genes between 0.25 and 0.75 inclusive.
    
    Parameters
    ----------
    save : a list of positive integers
        The time points at which snapshots of information were taken. Can be gathered by observing the file names.
    all_my_data : a list of master dictionaries 
        Creates a list to read in and store the master dictionary from each saved time point in a particular simulation. See "all_dict" in the function "run" above.
    """
    # define some variables from the last time point
    df = all_my_data[-1]["array"]
    p2i = all_my_data[-1]["p_to_i"]
    
    # determine which lineage is the most common
    alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
    genes = copy.copy(stats.mode(df[alive, p2i["gene_0"]:], axis=1)[0])
    genes = genes.flatten().astype(int)
    genes[genes%2==1] += 1
    genes = genes/2
    which = stats.mode(genes)[0][0]
    
    # make all the empty lists
    smell_value = []
    smell_value_std = []
    smell_avg_value = []
    smell_avg_value_std = []
    
    # loop through every time point
    for i in range(len(save)):
        # define some variables for the current time point
        df = all_my_data[i]["array"]
        p2i = all_my_data[i]["p_to_i"]
        
        # get the list of lineages and pick the worms that match the winning lineage
        alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
        genes = copy.copy(stats.mode(df[alive, p2i["gene_0"]:], axis=1)[0])
        genes = genes.flatten().astype(int)
        genes[genes%2==1] += 1
        genes = genes/2
        these = alive[genes==which]
        
        # find their average smell gene value and standard deviation
        smell_value.append(np.mean((df[these,p2i["smell_1"]] + df[these,p2i["smell_2"]])/2))
        smell_value_std.append(np.std((df[these,p2i["smell_1"]] + df[these,p2i["smell_2"]])/2))
        
        # find the average smell gene and standard deviation for the entire population
        smell_avg_value.append(np.mean((df[alive,p2i["smell_1"]] + df[alive,p2i["smell_2"]])/2))
        smell_avg_value_std.append(np.std((df[alive,p2i["smell_1"]] + df[alive,p2i["smell_2"]])/2))

    plt.errorbar(save, smell_value, yerr=smell_value_std, marker="o", color="b", label="Winning Lineage " + str(int(which)))
    plt.errorbar(save, smell_avg_value, yerr=smell_avg_value_std, marker="o", color="r", label="Population Average")
    plt.text(save[-1], smell_value[-1]-0.03, str(np.round(smell_value[-1],2)), ha="center", color="b")
    plt.text(save[-1], smell_avg_value[-1]+0.02, str(np.round(smell_avg_value[-1],2)), ha="center", color="r")
    plt.ylim(0.25,0.75)
    plt.axhline(y=0.5, color="black")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Average Value of Travel Direction Gene")
    plt.legend()

# Fraction of Mutants
def mutation(save, all_my_data):
    """ Create a line graph showing the fraction of mutant dauer genes (blue) and travel direction genes (red) in the population over time.
    Note that this only compares the genes of living worms at each time point to the genes of worms in the original population.
    
    Parameters
    ----------
    save : a list of positive integers
        The time points at which snapshots of information were taken. Can be gathered by observing the file names.
    all_my_data : a list of master dictionaries 
        Creates a list to read in and store the master dictionary from each saved time point in a particular simulation. See "all_dict" in the function "run" above.
    """
    # define some variables from the first time point
    df = all_my_data[0]["array"]
    p2i = all_my_data[0]["p_to_i"]
    
    # find the initial dauer and smell genes
    alive = df[:,p2i["alive"]]==1
    original_d = np.concatenate((df[alive,p2i["dauer_1"]], df[alive,p2i["dauer_2"]]))
    original_s = np.concatenate((df[alive,p2i["smell_1"]], df[alive,p2i["smell_2"]]))
    
    # create some empty lists
    fraction_d = []
    fraction_s = []
    
    # loop through every time point
    for i in range(len(save)):
        # define some variables for the current time point
        df = all_my_data[i]["array"]
        p2i = all_my_data[i]["p_to_i"]
        
        # find the current dauer genes
        alive = df[:,p2i["alive"]]==1
        dauer = np.concatenate((df[alive,p2i["dauer_1"]], df[alive,p2i["dauer_2"]]))
        fraction_d.append(np.sum(~np.isin(dauer, original_d))/len(dauer))
        
        # find the current smell genes
        smell = np.concatenate((df[alive,p2i["smell_1"]], df[alive,p2i["smell_2"]]))
        fraction_s.append(np.sum(~np.isin(smell, original_s))/len(smell))
    
    plt.plot(save, fraction_d, "-o", color="b", label="Dauer Genes")
    plt.plot(save, fraction_s, "-o", color="r", label="Travel Genes")
    plt.text(save[-1], fraction_d[-1]+0.05, str(np.round(fraction_d[-1],2)), ha="center", color="b")
    plt.text(save[-1], fraction_s[-1]+0.05, str(np.round(fraction_s[-1],2)), ha="center", color="r")
    plt.ylim(0,1)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Fraction of Mutant Genes")
    plt.legend()

# Determine the Winning Lineages
def winner(save, all_my_data):
    """ Prints out a table listing some statistics for the winning lineage at each saved time point. Statistics include average dauer gene, standard deviation, and fraction of the population. 
    
    Parameters
    ----------
    save : a list of positive integers
        The time points at which snapshots of information were taken. Can be gathered by observing the file names.
    all_my_data : a list of master dictionaries 
        Creates a list to read in and store the master dictionary from each saved time point in a particular simulation. See "all_dict" in the function "run" above.
    """
    d = {"Time (hrs)":save, "Winning Line":[], "Avg Dauer Gene":[], "Std Dev":[], "Frac of Pop":[]}

    for i in range(len(save)):
        # define some variables from the current time point
        df = all_my_data[i]["array"]
        p2i = all_my_data[i]["p_to_i"]
        
        # determine which lineage is the most common
        alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
        genes = copy.copy(stats.mode(df[alive, p2i["gene_0"]:], axis=1)[0])
        genes = genes.flatten().astype(int)
        genes[genes%2==1] += 1
        genes = genes/2
        which = stats.mode(genes)[0][0]
        
        # add data to the dictionary
        d["Winning Line"].append(which.astype(int))
        avg_gene = np.mean((df[alive[genes==which], p2i["dauer_1"]] + df[alive[genes==which], p2i["dauer_2"]])/2)
        d["Avg Dauer Gene"].append(np.round(avg_gene,3))
        std_gene = np.std((df[alive[genes==which], p2i["dauer_1"]] + df[alive[genes==which], p2i["dauer_2"]])/2)
        d["Std Dev"].append(np.round(std_gene,3))
        frac = np.sum(genes==which)/len(genes)
        d["Frac of Pop"].append(np.round(frac,3))
    
    df = pd.DataFrame(data=d)
    print(df)

# Lineage Tracking
def line_track(var):
    """ Create a line graph showing the number of worms in each genetic lineage over time on a high resolution time scale. Only labels the lineages remaining at the last time point.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
    # read in the file
    f = open("lineage_tracking.txt", "r")
    f1 = f.readlines()
    f.close()
    
    # create the columns
    cols = [i for i in np.arange(var["pop_size"])]
    cols = ["Time"] + cols
    
    # create a dataframe
    df = pd.DataFrame(columns=cols)
    for i in range(len(f1)):
        data = list(map(float, f1[i].split(" ")))
        if len(data) < df.shape[1]:
            df.loc[len(df)] = np.concatenate((data, np.zeros(df.shape[1]-len(data))*np.nan))
        else:
            df.loc[len(df)] = data
    df[df==0]=np.nan

    # create the plot
    for i in range(df.shape[1]-1):
        if i in np.where(df.loc[df.shape[0]-1,0:].notna())[0]:
            plt.plot(df["Time"], df[i], "-", label=i)
        else:
            plt.plot(df["Time"], df[i], "-")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Number of Worms in Lineage")
    plt.title("Genetic Lineages")
    plt.legend(title="Last Line(s)", bbox_to_anchor=(1, 1))

# Allele Tracking
def allele_track(var):
    """ Create a line graph showing the number of each allele (rounded to a whole number) present in the population over time on a high resolution time scale.
    Only labels the alleles remaining at the last time point.
    
    Parameters
    ----------
    var : a dictionary
        Lists all the user input parameters and a couple additional parameters.
    """
    # read in the file
    f = open("allele_tracking.txt", "r")
    f1 = f.readlines()
    f.close()
    
    # create the columns
    cols = [i for i in np.arange(var["dauer_gene"][1]+1)]
    cols = ["Time"] + cols
    
    # create a dataframe
    df = pd.DataFrame(columns=cols)
    for i in range(len(f1)):
        data = list(map(float, f1[i].split(" ")))
        if len(data) < df.shape[1]:
            df.loc[len(df)] = np.concatenate((data, np.zeros(df.shape[1]-len(data))*np.nan))
        elif len(data) > df.shape[1]:
            cols = np.arange(df.columns[-1]+1, df.columns[-1]+1+len(data) - df.shape[1])
            df2 = pd.DataFrame(np.zeros((len(df),len(data) - df.shape[1]))*np.nan, columns=cols)
            df = pd.concat([df, df2], axis=1)
            df.loc[len(df)] = data
        else:
            df.loc[len(df)] = data
    df[df==0]=np.nan
    
    # create the plot
    for i in range(df.shape[1]-1):
        if i in np.where(df.loc[df.shape[0]-1,0:].notna())[0]:
            plt.plot(df["Time"], df[i], "-", label=i)
        else:
            plt.plot(df["Time"], df[i], "-")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Number of Alleles in Population")
    plt.title("Dauer Alleles")
    plt.legend(title="Last Allele(s)", bbox_to_anchor=(1,1))

# Setup for Muller Diagram
def make_muller(pop_size, location):
    """ Put together and store a csv file with the information needed to create the muller plots in R.
    
    Parameters
    ----------
    pop_size : a positive integer
        The size of the initial population of worms, and thus, the number of unique lineages throughout the simulation.
    location : a file pathway string
        The path to the directory where you want to store the data for these muller plots. 
    """  
    # use this function in combination with muller_code.R
    # read in the file
    f = open("lineage_tracking.txt", "r")
    f1 = f.readlines()
    f.close()

    # create the basic data
    data_len = len(f1)*pop_size
    line = np.tile(np.arange(pop_size),len(f1))
    base_data = np.zeros((data_len, 3)).astype(int)
    base_data[:,1] = line

    # create the dataframe
    df = pd.DataFrame(base_data, index=np.arange(1,data_len+1), columns=["Time", "Identity", "Population"])
    count = 1
    for i in range(len(f1)):
        
        # define each column of data
        data = list(map(int, f1[i].split(" ")[1:]))
        time = np.repeat(int(f1[i].split(" ")[0]), pop_size)
        if len(data) < pop_size:
            data = np.concatenate((data, np.zeros((pop_size-len(data))))).astype(int)
        
        # insert the data into the dataframe
        end = pop_size + count - 1
        df.loc[count:end,"Time"] = time
        df.loc[count:end,"Population"] = data
        count += pop_size
    
    # save the dataframe in a csv file
    os.chdir(location)
    df.to_csv("lineage_data.csv", index=False)

# Compare Parameters Between Runs of a Particular Experiment
def compare_para(exp_num, location):
    """ Put together and store an Excel file that is used to compare between input parameters across different simulations within the same experiment.
    Parameters will appear as the columns in the Excel file and will be removed if all values are consistent. Only parameters that are altered across simulations will remain in the file.
    
    Parameters
    ----------
    exp_num : a positive integer
        The number incidating for which experiment to compare parameters. This function will choose all simulations that match this number.
    location : a file pathway string
        The path to the directory where all of your simulations for the chosen experiment are stored. Will also save the Excel file here.
    """
    # pick out all the correct files from the experiments folder
    os.chdir(location)
    exp_list = [i for i in os.listdir() if i.split("_")[1] == str(exp_num)]
    
    # create the empty dataframe with all current parameters
    cols = list(run(1, save=[])["par"].keys())
    df = pd.DataFrame(columns=cols)

    # loop through each file from the experiment
    for i in range(len(exp_list)):
        os.chdir(location + exp_list[i])
        all_my_data = open_pickle(1)
        
        # define some variables
        keys = list(all_my_data["par"].keys())
        values = list(all_my_data["par"].values())
        
        # populate the dataframe
        if keys == cols:
            df.loc[i,:] = values
        else:
            for key,value in zip(keys,values):
                df.loc[i,key] = value

    # remove all columns with no unique values
    for col in cols:
        if len(np.unique(np.array(df.loc[:,col]))) == 1:
            df = df.drop(col, 1)

    # save the dataframe in an excel file
    os.chdir(location)
    df.to_excel("exp_" + str(exp_num) + ".xlsx", index=False)

# Combination Graph
def combine_results(exp_num, param, iteration, location):
    """ Plot the average dauer gene from the time point specified for each simulation run in the experiment of choice. Plotted against the parameter varied in that experiment.
    
    Parameters
    ----------
    exp_num : a positive integer
        The number incidating for which experiment to select. This function will choose all simulations that match this number.
    param : a string
        The paramter that was varied in the experiment chosen. Must match one of the parameters initially input into the function called "run" above.
    iteration : an integer
        Specifies the time point from which information will be retrieved. There must be a file saved with an information snapshot from that specific time point.
    location : a file pathway string
        The path to the directory where all of your simulations for the chosen experiment are stored.
    """
    # pick out all the correct files from the experiments folder
    os.chdir(location)
    exp_list = [i for i in os.listdir() if i.split("_")[1] == str(exp_num)]
    
    # create the empty lists
    param_value = []
    dauer_value = []
    std_value = []
    
    # loop through each file from the experiment
    for i in exp_list:
        os.chdir(location + i)
        all_my_data = open_pickle(iteration)
        
        # define some variables
        df = all_my_data["array"]
        p2i = all_my_data["p_to_i"]
        var = all_my_data["par"]
        
        # append values to the lists
        param_value.append(var[param])
        # find the average dauer gene and standard deviation for the entire population
        alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
        dauer_value.append(np.mean((df[alive,p2i["dauer_1"]] + df[alive,p2i["dauer_2"]])/2))
        std_value.append(np.std((df[alive,p2i["dauer_1"]] + df[alive,p2i["dauer_2"]])/2))
    
    # find and plot the line of best fit
    m, b = np.polyfit(np.array(param_value), np.array(dauer_value), 1)
    plt.plot(np.array(param_value), m*np.array(param_value) + b, color="black")    
    
    # calculate correlation
    r = stats.pearsonr(param_value, dauer_value)[0]
    r_sq = np.round(r**2, 3)
    
    # plot the data points
    plt.errorbar(param_value, dauer_value, yerr=std_value, marker="o", ls="none", color="tab:blue")
    plt.yticks([var["dauer_gene"][0]+3,var["dauer_gene"][1]-2], ["more\n likely\n" + str(var["dauer_gene"][0]+3), "less\n likely\n" + str(var["dauer_gene"][1]-2)])
    plt.ylim(var["dauer_gene"][1]+1,var["dauer_gene"][0])
    plt.title("All Runs of Experiment " + str(exp_num) + " at Time Point " + str(iteration))
    plt.xlabel("Parameter Varied : " + param)
    plt.ylabel("Average Likelihood of L2d/Dauer")
    plt.text(max(param_value), var["dauer_gene"][0]+3, "R² = " + str(r_sq), ha="right")
    plt.text(max(param_value), var["dauer_gene"][0]+5, "m = " + str(np.round(m,2)), ha="right")

# Combination Graph Over Time
def combine_results_over_time(exp_num, param, location):
    """ Plot the average dauer gene from each time point for each simulation run in the experiment of choice. Colored based on the parameter varied in that experiment.
    
    Parameters
    ----------
    exp_num : a positive integer
        The number incidating for which experiment to select. This function will choose all simulations that match this number.
    param : a string
        The paramter that was varied in the experiment chosen. Must match one of the parameters initially input into the function called "run" above.
    location : a file pathway string
        The path to the directory where all of your simulations for the chosen experiment are stored.
    """
    # pick out all the correct files from the experiments folder
    os.chdir(location)
    exp_list = [i for i in os.listdir() if i.split("_")[1] == str(exp_num)]
    
    # find all the values of the parameter
    param_value = []
    for i in exp_list:
        os.chdir(location + i)
        all_my_data = open_pickle(1)
        param_value.append(all_my_data["par"][param])
    
    # sort the experiments by parameter
    exp_list = np.array(exp_list)[np.argsort(param_value)]
    param_value.sort()
        
    # determine the color of each line using a colormap theme
    cm = pylab.get_cmap("winter")
    colors = np.array(param_value) - min(param_value)
    colors = colors/max(colors)
    colors = [cm(1.*i) for i in colors]
    
    # loop through each file from the experiment
    for i, j in zip(exp_list, range(len(param_value))):
        os.chdir(location + i)
        time_saved = [int(file.split("_")[-1].split(".")[0]) for file in os.listdir() if file.split("_")[0] == "all"]
        time_saved.sort()
        
        # create the empty lists
        dauer_value = []
        std_value = []
        
        for k in time_saved:
            all_my_data = open_pickle(k)
            
            # define some variables
            df = all_my_data["array"]
            p2i = all_my_data["p_to_i"]
            var = all_my_data["par"]
            
            # find the average dauer gene and standard deviation for the entire population
            alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
            dauer_value.append(np.mean((df[alive,p2i["dauer_1"]] + df[alive,p2i["dauer_2"]])/2))
            std_value.append(np.std((df[alive,p2i["dauer_1"]] + df[alive,p2i["dauer_2"]])/2))    
    
        # plot the data points
        if j in [np.argmax(param_value==i) for i in np.unique(param_value)]:
            plt.errorbar(time_saved, dauer_value, yerr=std_value, marker="o", label=round(var[param],3), color=colors[j])
        else:
            plt.errorbar(time_saved, dauer_value, yerr=std_value, marker="o", color=colors[j])
    
    # create the rest of the plot
    plt.yticks([var["dauer_gene"][0]+3,var["dauer_gene"][1]-2], ["more\n likely\n" + str(var["dauer_gene"][0]+3), "less\n likely\n" + str(var["dauer_gene"][1]-2)])
    plt.ylim(var["dauer_gene"][1]+1,var["dauer_gene"][0])
    plt.title("All Runs of Experiment " + str(exp_num))
    plt.xlabel("Time (hrs)")
    plt.ylabel("Average Likelihood of L2d/Dauer")
    plt.legend(title=param, bbox_to_anchor=(1,1))

# Combination Graph for Travel Direction
def combine_smell_results(exp_num, param, iteration, location):
    """ Plot the average travel direction gene from the time point specified for each simulation run in the experiment of choice. Plotted against the parameter varied in that experiment.
    
    Parameters
    ----------
    exp_num : a positive integer
        The number incidating for which experiment to select. This function will choose all simulations that match this number.
    param : a string
        The paramter that was varied in the experiment chosen. Must match one of the parameters initially input into the function called "run" above.
    iteration : an integer
        Specifies the time point from which information will be retrieved. There must be a file saved with an information snapshot from that specific time point.
    location : a file pathway string
        The path to the directory where all of your simulations for the chosen experiment are stored.
    """
    # pick out all the correct files from the experiments folder
    os.chdir(location)
    exp_list = [i for i in os.listdir() if i.split("_")[1] == str(exp_num)]
    
    # create the empty lists
    param_value = []
    smell_value = []
    std_value = []
    
    # loop through each file from the experiment
    for i in exp_list:
        os.chdir(location + i)
        all_my_data = open_pickle(iteration)
        
        # define some variables
        df = all_my_data["array"]
        p2i = all_my_data["p_to_i"]
        var = all_my_data["par"]
        
        # append values to the lists
        param_value.append(var[param])
        # find the average smell gene and standard deviation for the entire population
        alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
        smell_value.append(np.mean((df[alive,p2i["smell_1"]] + df[alive,p2i["smell_2"]])/2))
        std_value.append(np.std((df[alive,p2i["smell_1"]] + df[alive,p2i["smell_2"]])/2))
    
    # find and plot the line of best fit
    m, b = np.polyfit(np.array(param_value), np.array(smell_value), 1)
    plt.plot(np.array(param_value), m*np.array(param_value) + b, color="black")    
    
    # calculate correlation
    r = stats.pearsonr(param_value, smell_value)[0]
    r_sq = np.round(r**2, 3)
    
    # plot the data points
    plt.errorbar(param_value, smell_value, yerr=std_value, marker="o", ls="none", color="tab:blue")
    plt.yticks([0.28,0.5,0.72], ["nbrs\n 0.28", "equal\n 0.50", "food\n 0.72"])
    plt.ylim(0.25,0.75)
    plt.title("All Runs of Experiment " + str(exp_num) + " at Time Point " + str(iteration))
    plt.xlabel("Parameter Varied : " + param)
    plt.ylabel("Avg Weight of Factors for Travel Direction")
    plt.text(max(param_value), 0.72, "R² = " + str(r_sq), ha="right")
    plt.text(max(param_value), 0.7, "m = " + str(np.round(m,3)), ha="right")

# Make a Histogram of Smell Gene Results
def count_smell_results(location, which_exp=[4,5,6,7], iteration=30000):
    """ Create a histogram of the average travel direction gene from the time point specified for each simulation run in each experiment listed.
    Can be used to compare initial versus final distributions of travel direction genes (from all simulations) by changing the iteration parameter.
    
    Parameters
    ----------
    location : a file pathway string
        The path to the directory where all of your simulations for the chosen experiments are stored. Must all be in the same folder.
    which_exp : a list of positive integers
        The list of numbers incidating for which experiments to select. This function will choose all simulations in all experiments that match these numbers.
    iteration : an integer
        Specifies the time point from which information will be retrieved. There must be files saved with information snapshots from that specific time point.
    """
    # create the empty list
    smell_value = []
    
    # loop through the experiments
    for exp in which_exp:
        # pick out all the correct files from the experiments folder
        os.chdir(location)
        exp_list = [i for i in os.listdir() if i.split("_")[1] == str(exp)]
        
        # loop through each file from the experiment
        for i in exp_list:
            os.chdir(location + i)
            all_my_data = open_pickle(iteration)
            
            # define some variables
            df = all_my_data["array"]
            p2i = all_my_data["p_to_i"]
            
            # find the average smell gene for the entire population
            alive = np.array(np.where(df[:,p2i["alive"]]==1))[0]
            smell_value.append(np.mean((df[alive,p2i["smell_1"]] + df[alive,p2i["smell_2"]])/2))
    
    f = sns.displot(smell_value, kde=True)
    f.set_axis_labels("Avg Weight of Factors for Travel Direction", "Number of Experiments")
