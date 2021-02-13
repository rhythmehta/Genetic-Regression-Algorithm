#!/usr/bin/env python
# coding: utf-8

# # Genetic Regression Algorithm  

# In[1]:


### Genetic algorithm to fit a regression line of the form y=ax+b to a 2-variable dataset

#importing required libraries
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math


#loading the data
my_data_file = 'temp_data.npy' #loading numpy array file
data = np.load(my_data_file)   #reading data form the file loaded on line above

#parameters
initial_pop_size = 100  #initial number of individuals/population size
mutation_rate = 0.05    #speed of mutation process
num_generations = 10    #total number of generations
chromosome_length = 2   #length of chromosome (proposed solution to the problem)
num_survivors = 50      #number of fittest survivors after one generation terminates


# In[2]:


def A():
    #array of 5000 equally distributed samples, calculated over the inclusive interval of [-1, 80]
    gene_pool = np.linspace(-1,80,num = 5000)
    
    #assigning dimension of initial population
    dimensions = (initial_pop_size, chromosome_length)
    
    #random values with size of dimensions from the gene_pool without replacement
    return np.random.choice(gene_pool, size=dimensions, replace=False) 


# ### Function A
# - initializes the population of solutions for genetic algorithm, called parent generation with initial_pop_size agents, each with chromosome_length values, this is 
#   important as the evolution will happen within this population
# - generates a list of 5000 evenly spaced values ranging from -1 to 80 and selects initial_pop_size * chromosome length values as the initial population.
# - takes no inputs, the knowledge of initial population size and length of each solution is required, and outputs the NumPy array of initial population.

# In[3]:


def B(coefficients):
    k = len(data)                                           #length of training data array elements read
    tot = 0                                                 #to store total cost (residuals' sum of squares)
    for j in range(k):                                      #iteration for each xy pair
        y = coefficients[0] * data[j,0] + coefficients[1]   #calc result in y = ax + b form
        res = data[j,1] - y                                 #calc individual cost/ each residual point
        tot += res**2                                       #summing up total cost, squaring balances negative values
    return tot/k                                            #normalizing for each solution by calc average cost


# ### Function B
# - calculates the average cost (fitness value) on all training data which is helpful to evaluate parameter efficiency, we seek to minimize cost hence increasing fitness 
#   level. This follows the rule of evolution i.e. survival of the fittest, those with higher fitness level have higher chance of reproductive ability.
# - iterates over data to compare the expected value calculated with the x with model y = ax + b, then we square the cost and average it over length of data.
# - takes two coefficients as inputs in list form and outputs average cost i.e. the fitness

# In[4]:


def C():
    fitlist = []                                           #list to store fit individuals with better reproduction ability
    for x in range(len(current_pop)):                      #iterating over total population
        fitlist.append(np.array([x,B(current_pop[x])]))    #appending individual and its fitness to the fitlist
    return np.array(fitlist)                               #returns fit list of individuals


# ### Function C 
# - calculates the cost(fitness) of the total population and store to compare later whcih ones are more suitable
# - iterates over each data element of the current_pop and appends it to fitlist array
# - takes no input and outputs the current population's fitness list.

# In[5]:


def D():
    #creates `num_survivors//2` indices of population into NumPy array list without replacement
    random_selection = np.random.choice(range(len(fitness_vector)), num_survivors//2, replace=False)
    
    #finds index of least cost within the individuals of randomly selected indices chosen in above line
    best = np.argmin(fitness_vector[random_selection,1])
    
    #locates the index of fitness vector
    best_index = random_selection[best]
    
    #returns current population value yielding the best performance
    return current_pop[int(fitness_vector[best_index][0])]


# ### Function D
# - compares to find the best-performance yielding parameter of a random subset from current population to keep the population evolving.
# - selects random subset from all fitness values and finds the smallest value's index among this subset and locates the individual that it corresponds to
# - takes no input and outputs a 1X2 array which is the best-performing parameter from the obtained random subset.

# In[6]:


def E():
    duplicate_size = len(new_population) - len(survivors)                   #calc number of slots to be filled
    duplicate_survivors = np.zeros((duplicate_size, chromosome_length))     #intitialize array filled with zeros
    for x in range(chromosome_length):                                      #for each column in the survivor array
        duplicate_survivors[:, x] = np.repeat(survivors[:, x], 4, axis=0)   #duplicate column 4 times
        duplicate_survivors[:, x] = np.random.permutation(duplicate_survivors[:, x])  #array shuffling equivalent to mating
    return duplicate_survivors


# ### Function E
# - returns the shuffled 4-time duplication of the survivors which gives randomness when evolving the new population.
# - duplicates the survivors list four times and randomly shuffles it
# - takes no input and outputs the 4-time duplication shuffled NumPy array.

# In[7]:


def F(array):
    mutation_level = 0.1 #mutation desired within a generation
 
    #forming parents pairs to have crossover
    parents = array[np.random.choice(array.shape[0], (initial_pop_size, 2), replace=False), ]
    
    children = (parents[:, 0] + parents[:, 1])/2  #crossover creating children by calc averages of the parents
    
    #calc mutating vector using mutation rate and mutation level
    mutation_vector = 1 + np.concatenate(
 ((np.random.random((initial_pop_size//2, 2)) < mutation_rate/2) * mutation_level,
 (np.random.random(((initial_pop_size - initial_pop_size//2), 2)) < mutation_rate/2) * -mutation_level))
    #mutating the generated crossover
    return children * mutation_vector


# ### Function F
# - responsible for mutations/crossovers important for change in genes
# - averages parents for crossover children and uses mutation_level to change mutated child
# - takes array with population data and outputs mutated children

# In[8]:


########################################################################
# Start of main program

#creates the initial population by using the function A
current_pop = A()

#creates an empty array to store new population with the size five times of the survivors
new_population = np.zeros((num_survivors * 5, chromosome_length))

# Initiate history
best_cost_history = []
median_cost_history = [] 

# main loop
for i in range(num_generations):
    #creates an array that contains the fitness values of the elements
    fitness_vector = C()
    
    #creates a new array of survivors
    survivors = np.zeros((num_survivors, chromosome_length))
    
    #iterate to choose a survivor using function D
    for n in range(len(survivors)):
        survivors[n] = D()
    
    #create the new population using survivors and crossover of survivors using the function E()
    new_population[:len(survivors)] = survivors
    new_population[len(survivors):] = E()
    
    #perform mutation to maintain diversity
    new_population = F(new_population)
    
    #update the current population
    current_pop = new_population
    
    #empty out the new population array for the next generation
    new_population = np.zeros((num_survivors * 5, chromosome_length))

#define the array which contains fitness values of current population that is the second generation
fitness_vector = C()

#find the best solution by selecting the solution with (least square or) most fitness value
best_solution = current_pop[np.argmin(fitness_vector[:,1])]

print("The best solution is", best_solution)
print("with error equal to approximately", B(best_solution))


# - This main part of program connects the functions we defined above, by randomly selecting the initial population, then selecting survivors from each generations to move
#   further by calculating their fitness after which as the time passes the survivors undergo evolutionary steps like of crossover children generation and mutations which
#   further passes to next generation. Lastly, it outputs best solution in the population we obtain as parameters after all the iterations are completed.

# In[9]:


# scipy regression function to compare with the genetic algorithm's solutions
print(stats.linregress(data)[0:2])

