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
def tournament_selection(population, matrix, tournament_size=2):
    new_population = []
    for _ in range(tournament_size):
        random.shuffle(population)
        for i in range(0, len(population)-1, 2):
            ind1 = population[i]
            ind2 = population[i+1]
            inds_to_compete = np.array([ind1, ind2])

            fitness_vector = fitness_function(matrix, inds_to_compete)
            best_ind = inds_to_compete[np.argmin(fitness_vector)]
            new_population.append(best_ind)

    return np.array(new_population)


# Mutation function
def mutation(individuals, prob_mutation, num_colors):
    for individual in individuals:
        if random.uniform(0, 1) <= prob_mutation:
            position = random.randint(0, len(individual)-1)
            individual[position] = random.randint(1, num_colors)

    return individuals

def get_results(results, penalty, num_colors, generations, fitness_evaluations):
    mean_penalty = np.sum(penalty) / len(penalty)
    min_penalty = np.min(penalty)

    sorted_penalty = np.sort(penalty)
    n_low_5_percent = int(0.20 * len(penalty))
    lowest_5_percent_values = sorted_penalty[:n_low_5_percent]
    mean_low_5_percent = np.mean(lowest_5_percent_values)

    new_row = {'Mean Penalty': mean_penalty, 'Min Penalty':min_penalty, 'Low 20% penalty': mean_low_5_percent, 
               'Generation': generations, 'Number colors': num_colors, 'Fitness evaluations': fitness_evaluations}
    results = results._append(new_row, ignore_index = True)
    return results


###################################### MAIN PROGRAM  ######################################

def obtain_colours(matrix, population_size = 100, prob_mutation = 0.2, max_generations = 200,
                   crossover_type = 'uniform', crossover_prob = 0.5):
    num_colors = max(np.sum(matrix, axis=0))
    nodes_graph = matrix.shape[0]
    best_palette = None
    results = pd.DataFrame()

    start_time = time.time()

    total_generations = 0
    fitness_evaluations = 0


    while num_colors > 0:
        # Initialize the population
        population = np.random.randint(num_colors, size=(population_size, nodes_graph))
        generations = 0

        # Evaluate individuals
        penalty = fitness_function(matrix, population)

        while np.min(penalty) > 0 and generations < max_generations:

            # Tournament selection to choose the individuals for reproduction
            individuals_to_reproduce = tournament_selection(population, matrix)

            # Crossover individuals to obtain the offspring
            new_population = []
            for i in range(0, len(individuals_to_reproduce)-1, 2):
                child1, child2 = crossover(crossover_type, individuals_to_reproduce[i], individuals_to_reproduce[i+1], crossover_prob)
                new_population.extend([child1, child2])

            # Apply Mutation to the new population
            new_population = mutation(new_population, prob_mutation, num_colors,)
            population = new_population

            # Evaluate the new population
            penalty = fitness_function(matrix, population)

            fitness_evaluations += len(penalty)

            results = get_results(results, penalty, num_colors, total_generations, fitness_evaluations)

            generations += 1
            total_generations += 1
            
            
        # If a valid solution is found, reduce the number of colors
        if np.min(penalty) == 0:
            num_colors -= 1
            best_palette = population[np.argmin(penalty)]

            print(f"Number of colors: {len(np.unique(best_palette))}, Generation: {generations}")

        else:
            break # No valid solution with fewer colors, so stop

    execution_time = time.time() - start_time

    # Information for the report
    """report_data = {
        'num_colors': len(np.unique(best_palette)) if best_palette is not None else None,
        'generations': generations,
        'fitness_calls': fitness_call_counter[0],
        'execution_time': execution_time
    }"""
        
    return best_palette, results, execution_time



