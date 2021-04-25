# set the working directory
setwd("C:/Users/Rachel/Documents/Rachel/BS MS Program/muller_plots/")

# set the population size to be the universal parent
pop_size = 200

# load required packages
library("ggmuller")
library("readr")
library("plyr")
library("dplyr")
library("magrittr")
library("ggplot2")
library("tidyr")

# read in the data and define starting values
pop_data = read_csv("lineage_data.csv")
start_time = min(pop_data$Time)
start_pop_size = pop_data %>% filter(Time == start_time) %>% pull(Population) %>% sum(.)

# make the edges matrix and add the universal parent
edges = tibble(Parent=pop_size, Identity=unique(pop_data$Identity))
pop_data = rbind.fill(tibble(Time=start_time-0.01, Identity=pop_size, Population=start_pop_size), pop_data)

# create the muller matrix
muller_matrix = get_Muller_df(edges, pop_data)

# create and print the plots
mp = Muller_plot(muller_matrix, add_legend = F, xlab = "Time", ylab = "Proportion")
mpp = Muller_pop_plot(muller_matrix, add_legend = F, xlab = "Time", ylab = "Population size")
print(mp)
print(mpp)