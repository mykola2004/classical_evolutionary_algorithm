import numpy as np
import math
import random
import time

#Global variables
NUM_EPOCHS, NUM_GENES, NUM_INDIVIDUALS, NUM_SIGNIFICANT_FIGURES = 0, 0, 0, 0
FUNCTION_NAME = None
TYPE = None
HIGH = []
LOW = []

def calculate_lengths_genes():
    """
    the function calculates how much bits i-th gene of an individual should have
    """
    lengths_genes = []
    for i in range(NUM_GENES):
        lengths_genes.append(math.ceil(math.log((HIGH[i]-LOW[i])*(10**NUM_SIGNIFICANT_FIGURES[i])+1, 2)) )
    return lengths_genes

def set_globals(num_epochs, num_genes, num_individuals, num_significant_figures, function_name, type, high, low):
    """
    Setter for global variables
    """
    global NUM_EPOCHS, NUM_GENES, NUM_INDIVIDUALS, NUM_SIGNIFICANT_FIGURES, FUNCTION_NAME, TYPE, HIGH, LOW
    NUM_EPOCHS, NUM_GENES, NUM_INDIVIDUALS, NUM_SIGNIFICANT_FIGURES, FUNCTION_NAME, TYPE, HIGH, LOW = num_epochs, num_genes, num_individuals, num_significant_figures, function_name, type, high, low

#python represenation of mathematical fucntions used for application
def binary_to_decimal(sequence, index):
    m = len(sequence)
    decimal = int(''.join(str(bit) for bit in sequence), 2)
    decimal_number = LOW[index] + decimal*(HIGH[index]-LOW[index])/(2**m-1)
    return decimal_number

def sphere(x):
    return sum(binary_to_decimal(x[i], i)**2 for i in range(len(x)))

def hyperellipsoid(x):
    return sum((i+1) * binary_to_decimal(x[i], i)**2 for i in range(len(x)))

def schwefel(x):
    n = len(x)
    return 418.9829 * n - sum(binary_to_decimal(x[i], i) * math.sin(math.sqrt(abs(binary_to_decimal(x[i], i)))) for i in range(len(x)))

def ackley(x):
    n = len(x)
    sum1 = sum(binary_to_decimal(x[i], i)**2 for i in range(len(x)))
    sum2 = sum(math.cos(2 * math.pi * binary_to_decimal(x[i], i)) for i in range(len(x)))
    return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e

def michalewicz(x, m=10):
    return -sum(math.sin(xi) * (math.sin((i+1) * xi**2 / math.pi)) ** (2*m) for i, bits in enumerate(x) for xi in [binary_to_decimal(bits, i)])

def rastrigin(x):
    n = len(x)
    return 10 * n + sum(binary_to_decimal(x[i], i)**2 - 10 * math.cos(2 * math.pi * binary_to_decimal(x[i], i)) for i in range(len(x)))

def rosenbrock(x):
    return sum(100 * (binary_to_decimal(x[i+1], i+1) - binary_to_decimal(x[i], i)**2)**2 + (binary_to_decimal(x[i], i) - 1)**2 for i in range(len(x)-1))

def dejong3(x):
    return sum(math.floor(binary_to_decimal(x[i], i)) for i in range(len(x)))

#dictionary linking the name of a fucntion and its name in code
dict_functions = {"Sphere": sphere, "Hyperellipsoid": hyperellipsoid, "Schwefel": schwefel, "Ackley": ackley,  "Michalewicz": michalewicz, "Rastrigin": rastrigin, "Rosenbrock": rosenbrock, "Dejong3": dejong3}

def cross_onepoint(x1, x2):
    x1_copy, x2_copy = [i for i in x1],  [i for i in x2]
    length = len(x1)
    ran_index = np.random.randint(1,length)
    x1_copy[ran_index:], x2_copy[ran_index:] = x2_copy[ran_index:], x1_copy[ran_index:]
    return [x1_copy, x2_copy]

def cross_twopoint(x1, x2):
    x1_copy = [i for i in x1]
    x2_copy = [i for i in x2]
    length = len(x1)
    ran_index1 = np.random.randint(0,length-1)
    ran_index2 = np.random.randint(ran_index1+1,length)
    while ran_index1 == 0 and ran_index2 == length - 1:
        ran_index1 = np.random.randint(0,length-1)
        ran_index2 = np.random.randint(ran_index1+1,length)
    x1_copy[ran_index1:ran_index2+1], x2_copy[ran_index1:ran_index2+1] = x2_copy[ran_index1:ran_index2+1], x1_copy[ran_index1:ran_index2+1]
    return [x1_copy, x2_copy]

def cross_threepoint(x1, x2):
    x1_copy = [i for i in x1]
    x2_copy = [i for i in x2]
    length = len(x1)
    r1 = np.random.randint(0,length-2)
    r2 = np.random.randint(r1+1,length-1)
    r3 = np.random.randint(r2+1, length)
    while r2+1 == r3 and r1 == 0:
        r1 = np.random.randint(0,length-2)
        r2 = np.random.randint(r1+1,length-1)
        r3 = np.random.randint(r2+1, length)
    x1_copy[r1:r2+1], x2_copy[r1:r2+1] = x2_copy[r1:r2+1], x1_copy[r1:r2+1]
    x1_copy[r3:], x2_copy[r3:] = x2_copy[r3:], x1_copy[r3:]
    return [x1_copy, x2_copy]

def cross_uniform(x1, x2):
    p = 0.5
    length = len(x1)
    x1_copy = [i for i in x1]
    x2_copy = [i for i in x2]
    counter = 0
    while counter == 0:
        for i in range(length):
            if counter == length - 1:
                break
            alpha = np.random.uniform(0, 1)
            if alpha <= p:
                counter += 1
                x1_copy[i], x2_copy[i] = x2_copy[i], x1_copy[i]
    return [x1_copy, x2_copy]

def cross_grain(x1, x2):
    length = len(x1)
    x_new = [None]*length
    flag1, flag2 = True, True
    while flag1 or flag2:
        flag1, flag2 = True, True
        for i in range(length):
            alpha = np.random.uniform(0, 1)
            if alpha <= 0.5:
                x_new[i] = x1[i]
                flag1 = False
            else:
                x_new[i] = x2[i]
                flag2 = False
    return x_new

def cross_shuffle(x1, x2):
    x1_new = [i for i in x1]
    x2_new = [i for i in x2]
    random.shuffle(x1_new)
    while x1_new == x1:
        random.shuffle(x1_new)
    random.shuffle(x2_new)
    while x2_new == x2:
        random.shuffle(x2_new)
    return [x1_new, x2_new]

def cross_replacement(x1, x2):
    x1_new = [i for i in x1]
    x2_new = [i for i in x2]
    indexes = []
    for i in range(len(x1_new)):
        if x1_new[i] != x2_new[i]:
            indexes.append(i)
    if len(indexes) != 0:
        if len(indexes) == 1:
            index = indexes[0]
        else:
            index = random.choice(indexes)
            while index == 0:
                index = random.choice(indexes)
        x1_new[index:], x2_new[index:] = x2_new[index:], x1_new[index:]
    return [x1_new, x2_new]

def cross_devastating(x1, x2):
    x1_new = [i for i in x1]
    x2_new = [i for i in x2]

    diff_counter = 0
    indexes = []
    for i in range(len(x1_new)):
        if x1_new[i] != x2_new[i]:
            diff_counter+=1
            indexes.append(i)

    swaps_counter = 0
    for i in indexes:
        if swaps_counter >= int(round(diff_counter/2)):
            break
        probability = np.random.uniform(0, 1)
        if probability < 0.5:
            x1_new[i], x2_new[i] = x2_new[i], x1_new[i]
            swaps_counter+=1
    return [x1_new, x2_new]

cross_names = {"onepoint": cross_onepoint, "twopoint": cross_twopoint, "threepoint": cross_threepoint, "devastation": cross_devastating, "grain": cross_grain, "replacement": cross_replacement, "shuffle": cross_shuffle, "uniform": cross_uniform}

def mutation_edge(binary_seq):
    if binary_seq[-1] == 0: binary_seq[-1] = 1
    else: binary_seq[-1] = 0
    return binary_seq

def mutation_onepoint(binary_seq):
    index = np.random.randint(0, len(binary_seq))
    if binary_seq[index] == 0: binary_seq[index] = 1
    else: binary_seq[index] = 0
    return binary_seq

def mutation_twopoint(binary_seq):
    i1, i2 = np.random.randint(0, len(binary_seq)), np.random.randint(0, len(binary_seq))
    while i1 == i2:
        i2 = np.random.randint(0, len(binary_seq))
    if binary_seq[i1] == 0: binary_seq[i1] = 1
    else: binary_seq[i1] = 0
    if binary_seq[i2] == 0: binary_seq[i2] = 1
    else: binary_seq[i2] = 0
    return binary_seq

def inversion(binary_seq): 
    i1 = np.random.randint(0, len(binary_seq)-1)
    i2 = np.random.randint(0, len(binary_seq))
    while i1 == i2:
        i1 = np.random.randint(0, len(binary_seq)-1)
        i2 = np.random.randint(0, len(binary_seq))
    if i1 < i2:
        binary_seq[i1: i2+1] = binary_seq[i1: i2+1][::-1]
    else:
        binary_seq[i2: i1+1] = binary_seq[i2: i1+1][::-1]
    return binary_seq

class Individual:
    '''
    class representing one point in d dimensions, or an individual of a population
    it is an array that conisits of d binary arrays, each represents point in d-th dimension
    '''
    def __init__(self, number_of_genes, lengths_genes):
        self.genom = None
        self.number_of_genes = number_of_genes
        self.lengths_genes = lengths_genes

    def get_genom(self):
        return self.genom
    
    def set_genom(self, new_genom):
        self.genom = new_genom

    def to_decimal(self):
        return [binary_to_decimal(self.genom[i], i) for i in range(self.number_of_genes)]

    def initialize_individual(self):
        self.genom = []
        for i in range(self.number_of_genes):
            gen = [np.random.randint(0, 2) for _ in range(self.lengths_genes[i])]
            self.genom.append(gen)

    def cross(self, individual, type):
        son, daughter, son_genom, daughter_genom = Individual(self.number_of_genes, self.lengths_genes), Individual(self.number_of_genes, self.lengths_genes), [], []
        if type == "grain":
            for i in range(self.number_of_genes):
                son_genom_i = cross_grain(self.genom[i], individual.get_genom()[i])
                son_genom.append(son_genom_i)
            son.set_genom(son_genom)
            daughter = None
        elif type in ["onepoint", "twopoint", "threepoint", "devastation", "replacement", "shuffle", "uniform"]:
            for i in range(self.number_of_genes):
                son_genom_i, daughter_genom_i = cross_names[type](individual.get_genom()[i], self.get_genom()[i])
                son_genom.append(son_genom_i)
                daughter_genom.append(daughter_genom_i)
            son.set_genom(son_genom), daughter.set_genom(daughter_genom)
        else:
            raise Exception("No such crossout type!")

        return son, daughter
    
    def mutate(self, type, rate):
        for j in range(len(self.genom)):
            prob_mutation = np.random.uniform(0, 1)
            if prob_mutation <= rate: 
                if type == "edge":
                    self.genom[j] = mutation_edge(self.genom[j])
                elif type == "onepoint":
                    self.genom[j] = mutation_onepoint(self.genom[j])
                elif type == "twopoint":
                    self.genom[j] = mutation_twopoint(self.genom[j])
                else:
                    raise Exception("No such mutation technique")
    
    def invert(self, rate):
        for i in range(len(self.genom)):
            prob_inversion = np.random.uniform(0, 1)
            if prob_inversion <= rate: 
                self.genom[i] = inversion(self.genom[i])
    
class Population:
    '''
    class representing the population of individuals
    '''
    def __init__(self, number_of_individuals, number_of_genes, lengths_genes):
        self.individuals = None
        self.grades = None
        self.number_of_individuals = number_of_individuals
        self.number_of_genes = number_of_genes
        self.lengths_genes = lengths_genes

    def get_all_individuals(self):
        return self.individuals
    
    def get_individual_by_index(self, index):
        return self.individuals[index]
    
    def set_individuals(self, new_individuals):
        self.individuals = new_individuals

    def add_individual(self, individual):
        self.individuals.append(individual)
    
    def get_grades(self):
        return self.grades
    
    def get_best_individual_target(self):
        self.sort_population()
        return self.individuals[0], self.grades[0]
    
    def set_number_individuals(self, n):
        self.number_of_individuals = n
    
    def initialize_population(self):
        self.individuals, self.grades = [], []
        for _ in range(self.number_of_individuals):
            individual = Individual(self.number_of_genes, self.lengths_genes)
            individual.initialize_individual()
            self.individuals.append(individual)

    def evaluate_grades(self): #population is array of binary arrays [ [1, 0, 0, 1], [0, 1, 0, 1], []]
        try:
            if TYPE == "min": self.grades = [  1 *dict_functions[FUNCTION_NAME](individual.get_genom()) for individual in self.individuals]
            if TYPE == "max": self.grades = [(-1)*dict_functions[FUNCTION_NAME](individual.get_genom()) for individual in self.individuals]
        except:
            raise Exception("Error: The task is not minimization, neither maximization")

    def sort_population(self): #the best individuals have the smallest target and placed eventually at the beginning
        self.evaluate_grades()
        sorted_grades_population = sorted(zip(self.grades, self.individuals), key=lambda x: x[0])
        self.grades, self.individuals = [i for i, _ in sorted_grades_population], [j for _, j in sorted_grades_population]

    def grade_competitors(self, individuals): #population is array of binary arrays [ [1, 0, 0, 1], [0, 1, 0, 1], []]
        try:
            if TYPE == "min": return [  1 *dict_functions[FUNCTION_NAME](individual.get_genom()) for individual in individuals]
            if TYPE == "max": return [(-1)*dict_functions[FUNCTION_NAME](individual.get_genom()) for individual in individuals]
        except:
            raise Exception("Error: The task is not minimization, neither maximization")       

    def sort_competitors(self, individuals): #the best individuals have the smallest target and placed eventually at the beginning
        grades = self.grade_competitors(individuals)
        sorted_grades_individuals = sorted(zip(grades, individuals), key=lambda x: x[0])
        return [j for _, j in sorted_grades_individuals]   
    
    def select_individuals(self, type, **kargs):
        if type in ["best", "worst", "roulette", "random", "ranking"]:
            percentage_keep = kargs["percentage_keep"]
            if type == "best": return self.select_best(percentage_keep)
            if type == "worst": return self.select_worst(percentage_keep)
            if type == "roulette": return self.select_roulette(percentage_keep)
            if type == "random": return self.select_random(percentage_keep)
            if type == "ranking": return self.select_ranking(percentage_keep)

        if type == "tournament":
            number_tournaments, k = kargs["number_tournaments"], kargs["k"]
            return self.select_tournament(number_tournaments, k)
        raise Exception("No such selection technique!")

    def select_best(self, percentage):
        self.sort_population()
        return self.individuals[0:int(round(self.number_of_individuals*percentage))]

    def select_worst(self, percentage):
        self.sort_population()
        return self.individuals[int(round(self.number_of_individuals-self.number_of_individuals*percentage)):]

    def select_random(self, percentage):
        selected = []
        for _ in range(int(round(self.number_of_individuals*percentage))):
            index = np.random.randint(0, int(self.number_of_individuals))
            selected.append(self.individuals[index])
        return selected

    def select_ranking(self, percentage):
        self.sort_population()
        extended_population = []
        for i in range(self.number_of_individuals):
            for _ in range(self.number_of_individuals - i):
                extended_population.append(self.individuals[i])
        chosen_population = random.sample(extended_population, int(math.floor(self.number_of_individuals*percentage)))
        return chosen_population

    def select_roulette(self, percentage):
        self.evaluate_grades()
        for i in range(self.number_of_individuals):
            if self.grades[i] < 0.000001: self.grades[i] = 0.00001 
        grades_inverted = [1/i for i in self.grades]
        sum = np.sum(grades_inverted)
        probabilities = [i/sum for i in grades_inverted]
        cumulative_probabilities = np.cumsum(probabilities).tolist()
        cumulative_probabilities.insert(0, 0)
        number_of_trials = int(round(self.number_of_individuals*percentage))
        chosen = []
        for _ in range(number_of_trials):
            p = np.random.uniform(0,1)
            for j in range(len(cumulative_probabilities)-1): 
                if (p >= cumulative_probabilities[j]) and (p < cumulative_probabilities[j+1]):
                    chosen.append(self.individuals[j])
        return chosen

    def select_tournament(self, number_tournaments, k):
        selected = []
        for _ in range(number_tournaments):
            competitors = random.sample(self.individuals, k)
            sorted_competitors = self.sort_competitors(competitors)
            selected.append(sorted_competitors[0])
        return selected
    
    def mutate(self, type, rate):
        for individual in self.individuals:
            individual.mutate(type, rate)

    def invert(self, rate):
        for individual in self.individuals:
            individual.invert(rate)

def main(function_name, type, number_of_genes, num_individuals, num_epochs, select_type, low, high, num_significant_figures, mutation_rate, inversion_rate, cross_type, mutation_type, elitarism=True, **conf):
    set_globals(num_epochs, number_of_genes, num_individuals, num_significant_figures, function_name, type, high, low) #setting up global varibales
    lengths_genes = calculate_lengths_genes() #calculating lengthes of each gene
    population = Population(NUM_INDIVIDUALS, NUM_GENES, lengths_genes) #creating population
    population.initialize_population() #initialization of population

    iterations_times, best_target_values, best_individual, best_decimal = [], [], None, None #variables for logging

    for epoch in range(NUM_EPOCHS): #main loop (from 1 to number of epochs)
        print(f"------------------------EPOCH NUMBER:{epoch}----------------------")
        print("SIZE POPULATION: ", population.number_of_individuals)
        time0 = time.perf_counter()

        best_individual, best_target = population.get_best_individual_target() #get the best individual with correposding target function value

        if type == "max": best_target = (-1)*best_target
        best_target_values.append(best_target)

        best_decimal = best_individual.to_decimal()
        print(f"______f({best_decimal}): {best_target}")

        selected_individuals = population.select_individuals(select_type, **conf) #select individuals

        new_individuals = []
        counter_new_individuals = 0
        if elitarism: limit_individuals = NUM_INDIVIDUALS - 1 #one individual will be kept from the previous generation
        else: limit_individuals = NUM_INDIVIDUALS

        while counter_new_individuals < limit_individuals: 
            index_mother, index_father = np.random.randint(0, len(selected_individuals)), np.random.randint(0, len(selected_individuals))
            while index_mother == index_father:
                index_father = np.random.randint(0, len(selected_individuals))
            #mother, father = population.get_individual_by_index(index_mother), population.get_individual_by_index(index_father) 
            mother, father = selected_individuals[index_mother], selected_individuals[index_father]
            son, daughter = father.cross(mother, cross_type)
            new_individuals.append(son)
            counter_new_individuals+=1
            if daughter != None and counter_new_individuals < limit_individuals:
                new_individuals.append(daughter)
                counter_new_individuals+=1

        population.set_individuals(new_individuals)
        population.set_number_individuals(counter_new_individuals)
        population.mutate(mutation_type, mutation_rate)
        population.invert(inversion_rate)

        if elitarism: 
            population.add_individual(best_individual)
            population.set_number_individuals(counter_new_individuals+1)
        time1 = time.perf_counter()
        iterations_times.append(time1-time0)

    print(":::::::::::::::::::::Coordinates of found optimal argument: ", best_decimal)

    return [best_decimal, best_target_values[-1], best_target_values, iterations_times]

#number_of_genes = 2
#num_individuals=50
##num_epochs = 500
##select_type = "best"
#low = [-5.12,-5.12]
#high = [5.12, 5.12]
#num_significant_figures = [6, 6]
#mutation_rate = 0.1
#inversion_rate = 0.1
#cross_type = "twopoint"
#mutation_type = "twopoint"
#elitarism = True
#main("Rastrigin", "max", number_of_genes, num_individuals, num_epochs, select_type, low, high, num_significant_figures, mutation_rate, inversion_rate, cross_type, mutation_type, elitarism, percentage_keep=0.2)