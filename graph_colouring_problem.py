import numpy as np
import random
import time

############# GENETIC ALGORITHM FUNCTIONS #############

# Create adjacent matrix of the map
def create_adjacent_matrix(df):
    nodes = len(df)
    adjacency_matrix = np.zeros((nodes, nodes), dtype=int)

    # Vectorized intersection checks using GeoPandas/NumPy
    for i in range(nodes):
        adjacency_matrix[i] = df.geometry.intersects(df.geometry[i]).astype(int)
        adjacency_matrix[i, i] = 0  # Remove self-adjacency

    return adjacency_matrix


# Define the fitness function to compute the penalty
def fitness_function(matrix, individuals, fitness_call_counter):
    # Directly sum conflicts for adjacent nodes
    penalty = np.zeros(len(individuals), dtype=int)
    for idx, individual in enumerate(individuals):
        adjacency_conflicts = matrix * (individual == individual[:, None])
        penalty[idx] = np.sum(adjacency_conflicts) // 2  # Divide by 2 to avoid double counting
    fitness_call_counter[0] += len(individuals)  # Count fitness calls
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


# Tournament selection function
def tournament_selection(population, matrix, fitness_call_counter, tournament_size=2):
    new_population = []
    for _ in range(tournament_size):
        random.shuffle(population)
        for i in range(0, len(population)-1, 2):
            ind1 = population[i]
            ind2 = population[i+1]
            inds_to_compete = np.array([ind1, ind2])

            fitness_vector = fitness_function(matrix, inds_to_compete, fitness_call_counter)
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


###################################### MAIN PROGRAM  ######################################

def obtain_colours(matrix):
    population_size = 100
    num_colors = max(np.sum(matrix, axis=0))
    nodes_graph = matrix.shape[0]
    best_palette = None
    prob_mutation = 0.2
    max_generations = 200
    fitness_call_counter = [0]  # A list to count fitness function calls (mutable object)

    start_time = time.time()

    while num_colors > 0:
        # Initialize the population
        population = np.random.randint(num_colors, size=(population_size, nodes_graph))
        generations = 0

        # Evaluate individuals
        penalty = fitness_function(matrix, population, fitness_call_counter)

        while np.min(penalty) > 0 and generations < max_generations:

            # Tournament selection to choose the individuals for reproduction
            individuals_to_reproduce = tournament_selection(population, matrix, fitness_call_counter)

            # Crossover individuals to obtain the offspring
            new_population = []
            for i in range(0, len(individuals_to_reproduce)-1, 2):
                child1, child2 = uniform_crossover(individuals_to_reproduce[i], individuals_to_reproduce[i+1])
                new_population.extend([child1, child2])

            # Apply Mutation to the new population
            new_population = mutation(new_population, prob_mutation, num_colors)
            population = new_population

            # Evaluate the new population
            penalty = fitness_function(matrix, population, fitness_call_counter)

            generations += 1
            
        # If a valid solution is found, reduce the number of colors
        if np.min(penalty) == 0:
            num_colors -= 1
            best_palette = population[np.argmin(penalty)]

            print(f"Number of colors: {len(np.unique(best_palette))}, Generation: {generations}")

        else:
            break # No valid solution with fewer colors, so stop

    execution_time = time.time() - start_time

    # Information for the report
    report_data = {
        'num_colors': len(np.unique(best_palette)) if best_palette is not None else None,
        'generations': generations,
        'fitness_calls': fitness_call_counter[0],
        'execution_time': execution_time
    }
        
    return best_palette, report_data



