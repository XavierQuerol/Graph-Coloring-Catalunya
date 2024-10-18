import random

# Objective function to optimize: DECIDIR SI MINIMITZAR O MAXIMITZAR
def objective_function(x):
    return x ** 2

# Function to decode a binary chromosome into a real value within the given range
def decode_chromosome(chromosome, min_x, max_x, bits=10):
    integer_value = int(chromosome, 2)
    return min_x + (max_x - min_x) * integer_value / (2**bits - 1)

# Function to create a random chromosome (binary string of length 'bits')
def create_chromosome(bits=10):
    return ''.join(random.choice('01') for _ in range(bits))

# Tournament selection function
def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    return min(selected, key=lambda ind: ind['fitness'])

# One-point crossover function --> LA PODEM CANVIAR PER TWO-POINTS CROSSOVER TAMBÉ (o més, el que volguem)
def crossover(parent1, parent2, crossover_rate=0.9):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]
    return parent1

# Bitwise mutation function
def mutate(chromosome, mutation_rate=0.01):
    mutated = ''
    for bit in chromosome:
        if random.random() < mutation_rate:
            mutated += '1' if bit == '0' else '0'
        else:
            mutated += bit
    return mutated

# Function to evaluate the fitness of a population
def evaluate_population(population, min_x, max_x):
    for individual in population:
        individual['x'] = decode_chromosome(individual['chromosome'], min_x, max_x)
        individual['fitness'] = objective_function(individual['x'])


###################################### MAIN PROGRAM  ######################################
        
# GA parameters
population_size = 10
generations = 50
crossover_rate = 0.9
mutation_rate = 0.01
bits = 10  # Chromosome length
min_x = -10  # Lower bound of the search space
max_x = 10   # Upper bound of the search space

# STEP 1. Initialize the random population (list of dictionaries)
population = [{'chromosome': create_chromosome(bits)} for _ in range(population_size)]

# STEP 2. Evaluate the initial population
evaluate_population(population, min_x, max_x)

# Genetic Algorithm
for generation in range(generations):
    new_population = []

    # Create the new population
    while len(new_population) < population_size:
        # Tournament selection to choose chromosomes to reproduce
        parent1 = tournament_selection(population)['chromosome']
        parent2 = tournament_selection(population)['chromosome']

        # Crossover
        offspring = crossover(parent1, parent2, crossover_rate)

        # Mutation
        offspring = mutate(offspring, mutation_rate)

        # Add the offspring to the new population
        new_population.append({'chromosome': offspring})

    # Replace the old population with the new one
    population = new_population

    # Evaluate the new population
    evaluate_population(population, min_x, max_x)

    # Find the best individual of the current generation
    best_individual = min(population, key=lambda ind: ind['fitness']) # ho podem canviar a maximitzar
    print(f"Generation {generation + 1}: Best x = {best_individual['x']:.5f}, f(x) = {best_individual['fitness']:.5f}")

# Best solution found
best_individual = min(population, key=lambda ind: ind['fitness']) # ho podem canviar a maximitzar
print(f"\nBest solution: x = {best_individual['x']:.5f}, f(x) = {best_individual['fitness']:.5f}")
