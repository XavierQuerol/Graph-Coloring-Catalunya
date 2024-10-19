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

    return matriz_adyacencia

def fitness_function(matrix, individuals):
    num_individus, num_nodes = individuals.shape # Número de nodos
    penalty = np.zeros(num_individus, dtype=int)
    for k, individual in enumerate(individuals):
        # Recorrer la matriz de adyacencia para detectar los conflictos
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # j empieza desde i+1 para evitar contar dos veces el mismo par
                if matrix[i][j] == 1:  # Si i y j son adyacentes
                    if individual[i] == individual[j]:  # Si tienen el mismo color
                        penalty[k] += 1  # Sumar 1 a la penalización por conflicto de color
    
    return penalty # vector


# One-point crossover function --> LA PODEM CANVIAR PER TWO-POINTS CROSSOVER TAMBÉ (o més, el que vulguem)
def crossover(parent1, parent2):
    n = len(parent1)
    position = random.randint(2, n-2) 
    child1 = [] 
    child2 = [] 
    for i in range(position+1): 
        child1.append(parent1[i]) 
        child2.append(parent2[i]) 
    for i in range(position+1, n): 
        child1.append(parent2[i]) 
        child2.append(parent1[i]) 
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

matriu = create_adjacent_matrix()
print(matriu)

# Crear inidvidus Initial population
p = 20
nodes_graf = 7
k = max(np.sum(matriu, axis=0)) # colors = 5

individuals = np.random.randint(k, size=(p,nodes_graf))
print(individuals)


# ALGORITME GA
generations = 0
# Evaluate individuals
penalty = fitness_function(matriu, individuals)
print(penalty)
prob_mutation=0.6
num_colors = nodes_graf - 1

# fer un bucle per anar reduint el numero de colors
while np.min(penalty) > 0 and generations < 1:

    # Shuffle individuals
    random.shuffle(individuals)

    # Triar els que es reproduiran - tournament selection
    individuals_to_reproduce = tournament_selection(individuals, matriu)

    # Crossover individuals_to_reproduce to obtain the offspring
    new_population = []
    for i in range(0, len(individuals_to_reproduce)-1, 2):
        child1, child2 = crossover(individuals_to_reproduce[i], individuals_to_reproduce[i+1])
        new_population.extend([child1, child2])

    print(new_population)

    # Ja tenim la nova population
    # Ara fem Mutation
    individuals_mutated = mutation(individuals, prob_mutation, num_colors)
    print(individuals_mutated)

