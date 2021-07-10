### Simulating the Evolutionary Effects of Environmental and Genetic Variation on Life History in *Caenorhabditis*

#### Abstract
Nematodes such as *Caenorhabditis elegans*, *C. briggsae*, *C. remanei*, and *C. nigoni* are model organisms that primarily reside in rotting fruit and plant matter, feeding on
the bacteria that inhabit these degrading vegetation (Frézal & Félix, 2015). When conditions are poor, worms must decide whether to go into a larval stage called dauer (Avery,
2014). Entering dauer would allow the worms to survive for months; however, the risk of dying in dauer is high and this results in a tradeoff. I created an extensive model in
Python to simulate the population dynamics and decision-making strategies of worms and their responses to various environmental conditions. This model includes genes related to
both dauer and travel direction decision making strategies. Experiments showed significant evolution of both genes when there is stronger selection against worms in dauer, both
genes when there is higher frequency of food availability, the dauer gene only when the dauer genotype to phenotype mapping is altered, and neither gene when seasonality in terms
of environmental productivity is introduced. Stronger selection against worms in dauer led them to evolve a lower likelihood of dauer and a preference for traveling away from
neighbors. A higher frequency of food availability also led them to evolve a lower likelihood of dauer but a preference for traveling towards food. A higher genotype to phenotype
mapping value led them to evolve a lower likelihood of dauer as well. However, there may be some underlying patterns present in many of these experiments that require further
study.

#### Files
- The file called "vectorized_RG.py" contains the model as well as various different functions for graphing the resulting data.
- The file called "modified_RG.py" also contains the model, but this script is prepared to run in the terminal. It requires user input to run.
- The file called "Worms_Life_Model.ipynb" is used to graph the results of each simulation. Just change the location and it will create many figures.
- The file called "muller_code.R" creates a couple of muller plots, but is used in conjunction with one of the functions in Python and the data from the simulation.
