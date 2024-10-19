import numpy as np
import random


# Create adjacent matrix - comarques girona
def create_adjacent_matrix():
    comarcas = ["Alt Empordà", "Baix Empordà", "Pla de l'Estany", "Gironès", "Selva", "Garrotxa", "Ripollès"]

    # Matriz de adyacencia (inicialmente todos ceros)
    n = len(comarcas)
    matriz_adyacencia = np.zeros((n, n), dtype=int)

    # Definir las conexiones entre comarcas
    conexiones = {
        "Alt Empordà": ["Baix Empordà", "Pla de l'Estany", "Gironès", "Garrotxa"],
        "Baix Empordà": ["Alt Empordà", "Gironès", "Selva"],
        "Pla de l'Estany": ["Alt Empordà", "Gironès", "Garrotxa"],
        "Gironès": ["Alt Empordà", "Baix Empordà", "Pla de l'Estany", "Selva", "Garrotxa"],
        "Selva": ["Gironès", "Garrotxa", "Baix Empordà"],
        "Garrotxa": ["Alt Empordà", "Pla de l'Estany", "Selva", "Ripollès", "Gironès"],
        "Ripollès": ["Garrotxa"]
    }

    # Rellenar la matriz de adyacencia
    for comarca, vecinas in conexiones.items():
        i = comarcas.index(comarca)
        for vecina in vecinas:
            j = comarcas.index(vecina)
            matriz_adyacencia[i][j] = 1
            matriz_adyacencia[j][i] = 1  # Porque es un grafo no dirigido

    return matriz_adyacencia, comarcas

def create_adjacent_matrix2(df):
    nodes = len(df)
    adjacency_matrix = np.zeros((nodes, nodes), dtype=int)

    # Vectorized intersection checks using GeoPandas/NumPy
    for i in range(nodes):
        adjacency_matrix[i] = df.geometry.intersects(df.geometry[i]).astype(int)
        adjacency_matrix[i, i] = 0  # Remove self-adjacency

    return adjacency_matrix

def fitness_function(matrix, individuals):
    # Directly sum conflicts for adjacent nodes
    penalty = np.zeros(len(individuals), dtype=int)
    for idx, individual in enumerate(individuals):
        adjacency_conflicts = matrix * (individual == individual[:, None])
        penalty[idx] = np.sum(adjacency_conflicts) // 2  # Divide by 2 to avoid double counting
    return penalty


# One-point crossover function --> LA PODEM CANVIAR PER TWO-POINTS CROSSOVER TAMBÉ (o més, el que vulguem)
def crossover(parent1, parent2):
    # Simplified crossover with slicing
    n = len(parent1)
    position = random.randint(2, n-2)
    child1 = np.concatenate((parent1[:position+1], parent2[position+1:]))
    child2 = np.concatenate((parent2[:position+1], parent1[position+1:]))
    return child1, child2


# Tournament selection function
def tournament_selection(population, matrix):
    new_population = [] # winners
    for j in range(2):
        random.shuffle(population)
        for i in range(0, len(population)-1, 2):
            invidu1 = population[i]
            invidu2 = population[i+1]
            population_compete = np.array([invidu1, invidu2])

            fitness_vector = fitness_function(matrix, population_compete)
            best_ind = population_compete[np.argmin(fitness_vector)]
            new_population.append(best_ind)

    return np.array(new_population)

def mutation(individuals, prob_mutation, num_colors):
    for individual in individuals:
        if random.uniform(0, 1) <= prob_mutation:
            position = random.randint(0, len(individual)-1)
            individual[position] = random.randint(1, num_colors)

    return individuals





###################################### MAIN PROGRAM  ######################################

def obtain_colors(matriu):

    # Crear inidvidus Initial population
    p = 100
    k = max(np.sum(matriu, axis=0)) # colors = 5
    nodes_graf = matriu.shape[0]

    num_colors = k

    best_palette = 0

        # fer un bucle per anar reduint el numero de colors
    while num_colors > 0:

        population = np.random.randint(num_colors, size=(p,nodes_graf))

        # ALGORITME GA
        generations = 0
        # Evaluate individuals
        penalty = fitness_function(matriu, population)
        prob_mutation=0.6

        while np.min(penalty) > 0 and generations < 200:

            # Shuffle individuals
            random.shuffle(population)

            # Triar els que es reproduiran - tournament selection
            individuals_to_reproduce = tournament_selection(population, matriu)

            # Crossover individuals_to_reproduce to obtain the offspring
            new_population = []
            for i in range(0, len(individuals_to_reproduce)-1, 2):
                child1, child2 = crossover(individuals_to_reproduce[i], individuals_to_reproduce[i+1])
                new_population.extend([child1, child2])

            # Ja tenim la nova population
            # Ara fem Mutation
            new_population = mutation(new_population, prob_mutation, num_colors)


            population = new_population
            penalty = fitness_function(matriu, population)

            generations += 1
            if generations % 100 == 0:
                print(num_colors, generations)

        if np.min(penalty) == 0:
            num_colors -= 1
            best_palette = population[np.argmin(penalty)]
        else:
            break
        print(num_colors)
        
    return best_palette



