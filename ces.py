# -*- coding: utf-8 -*-
#
# Convex Search Algorithm
#
import random
import statistics as stats

###########################
### AUXILIARY FUNCTIONS ###
###########################

def all_equal(bunch):
    first = bunch[0]
    return all(x == first for x in bunch)

def any_greater_than(bunch, threshold):
    return any(x > threshold for x in bunch)

##################################
### PROBLEMS' FITNESS FUNCTION ###
##################################

### Leading-Ones

def leadingones_fitness(individual):
    for position in range(INDIVIDUAL_SIZE):
        if individual[position] == 0:
            break
    else:
        position += 1
    return position
def leadingones_avg_fitness():
    ### avg(n) is the sum of fitnesses of all sequences with leading ones,
    ### i.e. (2^n) - 1, divided by the number of sequences with leading ones,
    ### i.e. 2^(n - 1):
    ###
    ###   avg(n)  = (2^n) - 1 / 2^(n - 1)
    ###           = (2^n / 2^n-1) - (1 / 2^n-1)
    ###           = 2 - 2^(1 - n) .
    ###
    ### In the limit: lim n->+inf (2 - 2^(1 - n)) = 2 - 0 = 2 .
    return 2 - (2**(1 - INDIVIDUAL_SIZE))

### One-Max

def onemax_fitness(individual):
    return sum(individual)

def onemax_avg_fitness():
    return sum([0.5 for _ in range(INDIVIDUAL_SIZE)])

######################
### REPRESENTATION ###
######################

def create_ind():
    return [random.randint(0, 1) for _ in range(INDIVIDUAL_SIZE)]

def create_pop():
    return [create_ind() for _ in range(POPULATION_SIZE)]

def evaluate_pop(population):
    return [onemax_fitness(individual) for individual in population]

#################################
### SELECTION META-HEURISTICS ###
#################################

def select_better_than_worst(population, fitness_population):
    worst_fitness = min(fitness_population)
    return [individual
            for (individual, fitness) in zip(population, fitness_population)
            if fitness > worst_fitness]

def select_above_avg(population, fitness_population):
    avg_fitness = onemax_avg_fitness()
    return [individual
            for (individual, fitness) in zip(population, fitness_population)
            if fitness > avg_fitness]

#####################
### RECOMBINATION ###
#####################

def convex_recombination_pop(mating_pool):
    return [convex_recombination_ind(mating_pool)
            for _ in range(POPULATION_SIZE)]

def convex_recombination_ind(mating_pool):
    transposed_mating_pool = zip(*mating_pool)
    def recombine_column(col):
        return col[0] if all_equal(col) else random.randint(0, 1)
    return list(map(recombine_column, transposed_mating_pool))

#####################
### CONVEX SEARCH ###
#####################

def convex_search():
    gens = 0
    population = create_pop()
    fitness_population = evaluate_pop(population)
    while (not all_equal(population)) and (gens < MAX_GENERATIONS):
        mating_pool = None
        if any_greater_than(fitness_population, leadingones_avg_fitness()):
            mating_pool = select_above_avg(population, fitness_population)
        if not all_equal(fitness_population):
            mating_pool = select_better_than_worst(
                population,
                fitness_population
            )
        else:
            mating_pool = population
        population = convex_recombination_pop(mating_pool)
        fitness_population = evaluate_pop(population)
        gens += 1
    return fitness_population[0]

############
### MAIN ###
############

### (Corollary 9) Recommende population sizes for a given individual size,
### so that convex search optimises LeadingOnes in O(n log n).
###     - population size: 25, 40,  60,   75
###     - individual size: 10, 100, 1000, 10000

MAX_RUNS        = 500
MAX_GENERATIONS = 100
POPULATION_SIZE = 25
INDIVIDUAL_SIZE = 1000

def main():
    ### Settings
    print("Max runs: %d, Max gens.: %d, Pop. size: %d, Ind. size: %d" %
        (MAX_RUNS, MAX_GENERATIONS, POPULATION_SIZE, INDIVIDUAL_SIZE)
    )
    ### Start
    runs = 0
    fitnesses = []
    while (runs < MAX_RUNS):
        fit = convex_search()
        fitnesses.append(fit)
        runs += 1
        print("Runs: %d" % runs)
    ### Results
    print("Max: %d, Min: %d, Avg: %f, Stdev: %f" %
        (max(fitnesses),
         min(fitnesses),
         stats.mean(fitnesses),
         stats.stdev(fitnesses))
    )

if __name__ == '__main__':
    main()

