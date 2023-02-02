import pandas as pd
import os, pickle 
import networkx as nx
from constants import FOOD_WEB_URL

def base_initialization(): 
    """
    Basic initialization of food webs. 
    
    Requires: 
        - User should add the URL to their food web 
          excel datasheet in constants.py

    Effects: 
        - Read in data from the excel spreadsheet and 
          create instances of networkx graphs for 
          food webs of every location in the datasheet. 
          Specifically, every graph will be directed 
          (predator -> prey) and edge will be populated 
          all with a weight of 1. The graphs will NOT 
          have any node features. 
    """
    # load in the food web excel sheet
    xls = pd.ExcelFile(FOOD_WEB_URL)
    prey_predator = pd.read_excel(xls, "Predator-Prey", skiprows=8)
    species_location = pd.read_excel(xls, "PA Data")

    # drop unnecessary columns from prey_predator
    omit_cols = ["Notes on prey", "Order", "Family", "Genus", "Mass.g"]
    prey_predator.drop(omit_cols, axis=1, inplace=True)
    prey_predator.set_index("Prey Species", inplace=True)
    # NA entries exist only for self-references
    prey_predator.fillna(0, inplace=True)
    # NOTE: making assumption that the 3-levels of edge information all 
    # imply predator->prey relationships
    prey_predator.replace(to_replace=[2.0, 3.0], value=1.0, inplace=True) 

    # drop unnecessary columns from species location
    species_location.drop(["Order", "Family", "Genus"], axis = 1, inplace=True)
    species_location.set_index("Species", inplace=True)

    # produce a mapping of location_name -> networkx graph
    location2graph = {}
    locations = species_location.columns
    for location in locations: 
        striped_location = location.strip()
        # select only species that exist within a certain location
        mask = species_location.loc[:, location] == 1
        species = species_location[mask].index
        # the directed graph is generated using transposed 
        # adjacency matrix because the original matrix has predators across 
        # the columns and preys across the rows
        location2graph[striped_location] = nx.DiGraph(prey_predator.loc[species, species].T)

    # save the location_to_nx_graph mapping
    filename_pkl = os.path.join('data/processed/foodweb_loc2nxgraph.pkl')
    with open(filename_pkl, 'wb') as f:
        pickle.dump(location2graph, f)

def biome_initialization():
    """
    Retrieve biome information from the food webs
    """
    biome_df = pd.read_csv("data/Biome.csv")
    loc2biome = {}
    for name in biome_df.Community: 
        striped_name = name.strip()
        loc2biome[striped_name] = biome_df[biome_df.Community == name].WWF_MHTNAM.values[0].strip()
    # save the name2biome dict
    with open("data/processed/foodweb_loc2biome.pkl", 'wb') as f:
        pickle.dump(loc2biome, f)

if __name__ == "__main__": 
    base_initialization()
    biome_initialization()

    


