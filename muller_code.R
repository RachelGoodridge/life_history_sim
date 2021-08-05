# set the working directory to wherever you have stored the muller graph data
setwd("D:/Worms_Life_Sim/muller_plots/")

# set the population size to be the universal parent
pop_size = 200

# load required packages
suppressMessages(library("ggmuller"))
suppressMessages(library("readr"))
suppressMessages(library("plyr"))
suppressMessages(library("dplyr"))
suppressMessages(library("magrittr"))
suppressMessages(library("ggplot2"))
suppressMessages(library("tidyr"))

# read in the data and define starting values
pop_data = read_csv("lineage_data.csv", col_types=cols(Time=col_integer(),Identity=col_integer(),Population=col_integer()))
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