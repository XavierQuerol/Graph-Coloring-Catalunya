import numpy as np
import pandas as pd
import random
import time

############# GENETIC ALGORITHM FUNCTIONS #############

# Define the fitness function to compute the penalty
def fitness_function(matrix, individuals):
    # Directly sum conflicts for adjacent nodes
    penalty = np.zeros(len(individuals), dtype=int)
    for idx, individual in enumerate(individuals):
        adjacency_conflicts = matrix * (individual == individual[:, None])
        penalty[idx] = np.sum(adjacency_conflicts) // 2  # Divide by 2 to avoid double counting
    return penalty


# One-point crossover function
def one_point_crossover(parent1, parent2):
    n = len(parent1)
    position = random.randint(2, n-2)
    child1 = np.concatenate((parent1[:position+1], parent2[position+1:]))
    child2 = np.concatenate((parent2[:position+1], parent1[position+1:]))
    return child1, child2


# Two-point crossover function
def two_point_crossover(parent1, parent2):
    n = len(parent1)
    point1, point2 = sorted(random.sample(range(1, n-1), 2))
    child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
    return child1, child2


# Uniform crossover function
def uniform_crossover(parent1, parent2, crossover_prob=0.5):
    n = len(parent1)
    child1 = np.empty(n, dtype=parent1.dtype)
    child2 = np.empty(n, dtype=parent2.dtype)
    
    for i in range(n):
        if random.uniform(0, 1) < crossover_prob:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
        else:
            child1[i] = parent2[i]
            child2[i] = parent1[i] 

    return child1, child2


def crossover(crossover_type, parent1, parent2, prob):
    if crossover_type == 'uniform':
        child1, child2 = uniform_crossover(parent1, parent2, prob)
    elif crossover_type == 'one-point':
        child1, child2 = one_point_crossover(parent1, parent2)
    elif crossover_type == 'two-point':
        child1, child2 = two_point_crossover(parent1, parent2)
    return child1, child2


# Tournament selection function
def tournament_selection(population_dict, tournament_size=2):
    selected_population_dict = []
    for _ in range(tournament_size):
        random.shuffle(population_dict)
        for i in range(0, len(population_dict)-1, 2):
            ind1 = population_dict[i]
            ind2 = population_dict[i+1]
            
            best_ind = ind1 if ind1['penalty'] < ind2['penalty'] else ind2
            selected_population_dict.append(best_ind)

    return selected_population_dict


# Mutation function
def mutation(individuals, prob_mutation, num_colors):
    for individual in individuals:
        if random.uniform(0, 1) <= prob_mutation:
            position = random.randint(0, len(individual)-1)
            individual[position] = random.randint(1, num_colors)

    return individuals


def get_results(results, population_dict, num_colors, generations, fitness_evaluations):
    penalties = [ind['penalty'] for ind in population_dict]  # Accedemos directamente a la lista
    mean_penalty = np.mean(penalties)
    min_penalty = np.min(penalties)

    sorted_penalty = np.sort(penalties)
    n_low_20_percent = int(0.20 * len(penalties))
    lowest_20_percent_values = sorted_penalty[:n_low_20_percent]
    mean_low_20_percent = np.mean(lowest_20_percent_values)

    new_row = {
        'Mean Penalty': mean_penalty,
        'Min Penalty': min_penalty,
        'Low 20% penalty': mean_low_20_percent,
        'Generation': generations,
        'Number colors': num_colors,
        'Fitness evaluations': fitness_evaluations
    }
    
    results = results._append(new_row, ignore_index=True)
    return results


###################################### MAIN PROGRAM  ######################################

def obtain_colours(matrix, population_size = 100, prob_mutation = 0.2, max_generations = 200,
                   crossover_type = 'uniform', crossover_prob = 0.5, elitism_size = 5):
    num_colors = max(np.sum(matrix, axis=0))
    nodes_graph = matrix.shape[0]
    best_palette = None
    results = pd.DataFrame()

    start_time = time.time()

    total_generations = 0
    fitness_evaluations = 0

    while num_colors > 0:
        generations = 0

        # Initialize the population
        population = np.random.randint(num_colors, size=(population_size, nodes_graph))
        
        # Evaluate individuals
        penalty = fitness_function(matrix, population)
        
        # Create the dictionary with the individuals of the current population and their penalty
        population_dict = [
            {'individual': individual, 'penalty': penalty_value}
            for individual, penalty_value in zip(population, penalty)
        ]

        while min(ind['penalty'] for ind in population_dict) > 0 and generations < max_generations:

            # Tournament selection to choose the individuals for reproduction
            individuals_to_reproduce_dict = tournament_selection(population_dict)

            # Crossover individuals to obtain the offspring
            new_population = []
            for i in range(0, len(individuals_to_reproduce_dict)-1, 2):
                child1, child2 = crossover(crossover_type, individuals_to_reproduce_dict[i]['individual'], individuals_to_reproduce_dict[i+1]['individual'], crossover_prob)
                new_population.extend([child1, child2])

            # Apply Mutation to the new population
            new_population = mutation(new_population, prob_mutation, num_colors,)

            # Evaluate new population
            penalty = fitness_function(matrix, new_population)

            new_population_dict = [
                {'individual': individual, 'penalty': penalty_value}
                for individual, penalty_value in zip(new_population, penalty)
            ]

            # Apply elitism: Preserve the best 'elitism_size' individuals
            sorted_population_dict = sorted(population_dict, key=lambda x: x['penalty'])
            best_individuals = sorted_population_dict[:elitism_size]

            # Add the best individuals back into the new population
            new_population_dict = best_individuals + new_population_dict

            # Update the population
            population_dict = new_population_dict

            # Shuffle the population
            random.shuffle(population_dict)

            fitness_evaluations += len(population_dict)

            results = get_results(results, population_dict, num_colors, total_generations, fitness_evaluations)

            generations += 1
            total_generations += 1
            
            
        # If a valid solution is found, reduce the number of colors
        if np.min([ind['penalty'] for ind in population_dict]) == 0:
            num_colors -= 1
            best_palette = min(population_dict, key=lambda x: x['penalty'])['individual']

            print(f"Number of colors: {len(np.unique(best_palette))}, Generation: {generations}")

        else:
            break # No valid solution with fewer colors, so stop

    execution_time = time.time() - start_time
        
    return best_palette, results, execution_time



