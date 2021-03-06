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
    return any(x >= threshold for x in bunch)


### Using any_two_greater_than instead of any_greater_than may be preferable
### because it avoids constructing a mating pool of individuals above average
### containing only a single individidual. Doing recombination only on one
### individual can only lead to the same individual, and this might accelerate
### premature convergence. Whereas if we have a mating pool consisting of at
### least two different individuals, the offspring are more likely to be
### different than parents.
def any_two_greater_than(bunch, threshold):
    return len(set([x for x in bunch if x >= threshold])) > 2

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
    return [PROBLEM_FITNESS(individual) for individual in population]

#################################
### SELECTION META-HEURISTICS ###
#################################

def select_better_than_worst(population, fitness_population):
    worst_fitness = min(fitness_population)
    return [individual
            for (individual, fitness) in zip(population, fitness_population)
            if fitness > worst_fitness]

def select_above_avg(population, fitness_population):
    avg_fitness = stats.mean(fitness_population)
    return [individual
            for (individual, fitness) in zip(population, fitness_population)
            if fitness >= avg_fitness]

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
        mating_pool = population
        if not all_equal(fitness_population):
            mating_pool = select_better_than_worst(
                population,
                fitness_population
            )
        population = convex_recombination_pop(mating_pool)
        fitness_population = evaluate_pop(population)
        gens += 1
    return (fitness_population[0], gens)

def convex_search2():
    gens = 0
    population = create_pop()
    fitness_population = evaluate_pop(population)
    while (not all_equal(population)) and (gens < MAX_GENERATIONS):
        mating_pool = population
        avg_fitness_pop = stats.mean(fitness_population)
        if any_two_greater_than(fitness_population, avg_fitness_pop):
            mating_pool = select_above_avg(population, fitness_population)
        elif not all_equal(fitness_population):
            mating_pool = select_better_than_worst(
                population,
                fitness_population
            )
        population = convex_recombination_pop(mating_pool)
        fitness_population = evaluate_pop(population)
        gens += 1
    return (fitness_population[0], gens)




############
### MAIN ###
############

### (Corollary 9) Recommended population sizes for a given individual size,
### so that "normal" convex search optimises LeadingOnes in O(n log n).
###     - population size: 25, 40,  60,   75
###     - individual size: 10, 100, 1000, 10000

SEARCH          = convex_search2
PROBLEM_FITNESS = leadingones_fitness
MAX_RUNS        = 300
MAX_GENERATIONS = 100
POPULATION_SIZE = 50
INDIVIDUAL_SIZE = 100

def main():
    ### Settings
    print("-------------------- SETTINGS --------------------")
    print("Search algorithm: %s" % SEARCH.__name__)
    print("Problem fitness function: %s" % PROBLEM_FITNESS.__name__)
    print("Max runs: %d, Max gens.: %d, Pop. size: %d, Ind. size: %d" %
        (MAX_RUNS, MAX_GENERATIONS, POPULATION_SIZE, INDIVIDUAL_SIZE)
    )
    ### Start
    print("-------------------- START --------------------")
    runs = 0
    fitnesses = []
    generations = []
    while (runs < MAX_RUNS):
        (fit, gens) = SEARCH()
        fitnesses.append(fit)
        generations.append(gens)
        runs += 1
        print("Runs: %d | fitness: %d, gens.: %d" % (runs, fit, gens))
    ### Results
    print("-------------------- SUMMARY --------------------")
    print("Fitness | Max: %d, Min: %d, Avg: %f, Median high: %f, Stdev: %f"
        % (max(fitnesses),
           min(fitnesses),
           stats.mean(fitnesses),
           stats.median_high(fitnesses),
           stats.stdev(fitnesses))
    )
    print("Generations | Max: %d, Min: %d, Avg: %f, Median low: %f, Stdev: %f"
        % (max(generations),
           min(generations),
           stats.mean(generations),
           stats.median_low(generations),
           stats.stdev(generations))
    )

if __name__ == '__main__':
    main()

