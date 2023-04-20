# Librerias
import pygad  # Libreria open-source para Algorítmos genéticos en python
import numpy as np        # Libreria para hacer operaciones matemáticas
import argparse
from scipy.optimize import root

lower_coefficient_value = 1
upper_coefficient_value = 256


# This was building from instructions cycles delay
def auxFunDelay(var, k):
    if k == 1:
        return 3*var[0]+6

    return (auxFunDelay(var, k-1)-2*k+3)*var[k-1]+2*k+4


def funDelay(var):
    '''
    var=[A, B, C .. ] from equations to delay
    return : Tbus or number os cicles of bus
    '''
    return auxFunDelay(var, var.size)


def auxVariableNumbers(TbusTimes, n):
    lowerVariableNumbers = np.ones(n, dtype=int)*lower_coefficient_value
    upperVariableNumbers = np.ones(n, dtype=int)*upper_coefficient_value

    if TbusTimes < funDelay(lowerVariableNumbers):
        return n-1
    elif TbusTimes >= funDelay(lowerVariableNumbers) and TbusTimes <= funDelay(upperVariableNumbers):
        return n

    return auxVariableNumbers(TbusTimes, n+1)


def variableNumbers(TbusTimes):
    '''
    Calculate how may variables we need
    '''
    return auxVariableNumbers(TbusTimes, 1)


def auxOptFunctionDelay(y, position, coefficientFloat, TbusTimes):
    coefficientFloat[position] = y
    return TbusTimes-funDelay(coefficientFloat)


def optFunDelay(coefficient, TbusTimes):
    # To obtain exactly root
    coefficientFloat = coefficient.astype(float)
    coefficientSort = np.sort(coefficient)
    positionArrayBeforeSort = np.zeros_like(coefficient)

    # know the positions of lower until upper
    for i in range(coefficient.size):
        index = np.nonzero(coefficient == coefficientSort[i])[0][0]
        positionArrayBeforeSort[i] = index

    # print(f"positionArrayBeforeSort: {positionArrayBeforeSort}")
    for pos in positionArrayBeforeSort:
        sol = root(auxOptFunctionDelay, coefficientFloat[pos],
                   args=(pos, coefficientFloat, TbusTimes))
        print(f"sol : {sol.x}")

        if sol.x > upper_coefficient_value:
            coefficientFloat[pos] = upper_coefficient_value
        elif sol.x < lower_coefficient_value:
            coefficientFloat[pos] = lower_coefficient_value
        else:
            if pos == positionArrayBeforeSort[-1]:
                coefficientFloat[pos] = np.floor(sol.x)
            else:
                coefficientFloat[pos] = np.around(sol.x)
    return coefficientFloat


# https://www.phind.com/
parser = argparse.ArgumentParser(
    description="Parameters to calculate the delay")
parser.add_argument("fosc", type=float,
                    help="The frequency clock in Hz")
parser.add_argument("delay", type=float,
                    help="Time that you need in seconds [s]")
args = parser.parse_args()

# Period for each instruction
Tbus = 4/args.fosc

# Periods of bus need to the delay
TbusTimes = int(np.round(args.delay/Tbus, 0))

# Hasta aquí está dando perfecto gracias a Dios


# Inicio de programación genética
# Función de aptitud la cual para cada solución
def fitness_func(solution, solution_idx):
    TbusTimesPredicted = funDelay(solution)
    fitness = 1.0 / (np.abs(TbusTimesPredicted - TbusTimes)+1e-3)
    if TbusTimesPredicted > TbusTimes:
        fitness = fitness*0.5
    return fitness

# Ahora se preparan los parámetros de PyGAD


fitness_function = fitness_func

sol_per_pop = 8
num_genes = variableNumbers(TbusTimes)

num_generations = 1000*(num_genes)
num_parents_mating = 4

init_range_low = lower_coefficient_value
init_range_high = upper_coefficient_value

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = (100/num_genes)

# Se crea una instancia de la clase pygad.GA

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       gene_type=int,
                       mutation_percent_genes=mutation_percent_genes)

# La optimización se inicia con el método run()

ga_instance.run()

# Para visualizar los resultados:
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
prediction = funDelay(solution)
print(f"Predicted output based on the best solution : {prediction} [# Tbus]")
print(f"Need - Predicted: {TbusTimes-prediction} [# Tbus]")
print(f"Final delay time : {prediction*Tbus} [s]")


# Optimization

print("\n\nOptimization\n")

optCoefficient = optFunDelay(solution, TbusTimes).astype(int)
prediction = funDelay(optCoefficient)
print(f"optCoefficient : {optCoefficient}")
print(f"Predicted: {prediction} [# Tbus]")
print(f"Need - Predicted: {TbusTimes-prediction} [# Tbus]")
print(f"Final delay time : {prediction*Tbus} [s]")
print(f"Relative error [%] :{(args.delay-prediction*Tbus)*100/args.delay}")

# Thank to
# https://www.phind.com
