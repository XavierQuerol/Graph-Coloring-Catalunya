from graph_coloring_problem import obtain_colours
from adjacent_matrix import create_adjacent_matrix
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def number_to_discrete_color(index, total_colors):
    cmap = plt.get_cmap('tab20b', total_colors)
    rgba = cmap(index / total_colors)
    return mcolors.to_hex(rgba)


def main():
    gdf = gpd.read_file('./COMARQUES/divisions-administratives-v2r1-comarques-250000-20240701.shp')
    gdf['NOM'] = gdf['NOMCOMAR']

    matrix = create_adjacent_matrix(gdf)

    best_population_size = 500
    best_prob_mutation = 0.05
    best_max_gen = 800
    best_crossover_type = 'uniform'
    best_palette, results, results2, execution_time_ = obtain_colours(matrix, population_size=best_population_size,
                                                        prob_mutation=best_prob_mutation, 
                                                        max_generations=best_max_gen,
                                                        crossover_type=best_crossover_type, 
                                                        crossover_prob=0.5)

    # Get the unique numbers from the list and assign each a color
    unique_numbers = list(set(best_palette))
    total_unique_colors = len(unique_numbers)

    # Create a dictionary mapping each unique number to a color
    number_color_mapping = {num: number_to_discrete_color(i, total_unique_colors) for i, num in enumerate(unique_numbers)}

    # Now map each comarca name to its corresponding color based on its number
    comarca_colors = {name: number_color_mapping[num] for name, num in zip(gdf['NOM'], best_palette)}

    # Create a new column for colors based on the comarca names
    gdf['color'] = gdf['NOM'].map(comarca_colors).fillna('#FFFFFF')

    # Plot the map
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    gdf.plot(ax=ax, color=gdf['color'], edgecolor='black')
    ax.set_title("Map of Catalonia by Comarques", fontsize=15)
    plt.axis('off')
    plt.show()

    print(f'The optimal number of colors is: {len(np.unique(best_palette))}')

    
    return best_palette


if __name__ == '__main__':
    main()