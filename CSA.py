from copy import deepcopy
import pandas as pd
import numpy as np
from ProblemClass import Problema, Solucao
import SolverFunctions as solver
from typing import List
from random import random, randint, uniform
from time import time
import BenchmarkFuntions as bf


def csa(cost_func, bounds, dimensions=10, pop_size=50, max_iter=200, pa=0.25, pc=0.5, beta=2):
	start_time = time()
	# Initialize population
	population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dimensions))
	fitness = np.apply_along_axis(cost_func, 1, population)

	# Record best cost and solution
	best_cost = np.min(fitness)
	best_solution = population[np.argmin(fitness)]

	# Main loop
	for t in range(max_iter):
		# Update position of each crow
		for i in range(pop_size):
			# Levy flight
			step_size = 1 / np.sqrt(t + 1)
			levy = np.random.standard_cauchy(size=dimensions)
			crow_pos = population[i] + step_size * levy / np.power(np.abs(levy), 1 / beta)

			# Boundary handling
			crow_pos = np.maximum(crow_pos, [bounds[0] for _ in range(dimensions)])
			crow_pos = np.minimum(crow_pos, [bounds[1] for _ in range(dimensions)])

			# Evaluate new position
			crow_fitness = cost_func(crow_pos)

			# Crow behavior
			if crow_fitness < fitness[i]:
				# Improvement: update crow position and forget
				population[i] = crow_pos
				fitness[i] = crow_fitness
				if crow_fitness < best_cost:
					best_cost = crow_fitness
					best_solution = crow_pos
			else:
				# Exploration: select one of the best crows and follow
				best_crows = population[fitness <= np.percentile(fitness, 100 - pa * 100)]
				selected_crow = best_crows[np.random.choice(len(best_crows))]
				population[i] = population[i] + pc * (selected_crow - population[i])

				# Boundary handling
				population[i] = np.maximum(population[i], [bounds[0] for _ in range(dimensions)])
				population[i] = np.minimum(population[i], [bounds[1] for _ in range(dimensions)])

				# Evaluate new position
				fitness[i] = cost_func(population[i])

				if fitness[i] < best_cost:
					best_cost = fitness[i]
					best_solution = population[i]

	print(f'\nTime Elapsed: {(time() - start_time):.2f}s')
	return best_solution, best_cost


def solve(population: List[Solucao], bounds, max_iter, pa=0.25, pc=0.5, beta=2):
	dimensions = population[0].MatrizSolucao.shape
	pop_size = len(population)
	# Define lower and upper bounds for each dimension
	lower_bounds = [bounds[0] for _ in range(dimensions[1])]
	upper_bounds = [bounds[1] for _ in range(dimensions[1])]

	start_time = time()
	# Initialize population

	population_fitness = np.array([sol.fitness for sol in population])
	# fitness = np.apply_along_axis(cost_func, 1, population)

	# Record best cost and solution
	best_fitness = np.min(population_fitness)
	best_solution = population[np.argmin(population_fitness)]

	resultados = [[0, best_fitness]]

	# Main loop
	for iter_num in range(1, max_iter+1):
		# Update position of each crow
		for i in range(pop_size):
			# Levy flight
			step_size = 1 / np.sqrt(iter_num)
			# levy = np.random.standard_cauchy(size=dimensions) # array that has a non-zero probability of generating very large values
			# val = np.round(step_size * levy / np.power(np.abs(levy), 1 / beta))

			rand_array = np.random.uniform(lower_bounds, upper_bounds, size=dimensions)  # array that has a non-zero probability of generating very large values
			val = np.round(step_size * rand_array / np.power(np.abs(rand_array), 1 / beta))

			crow_pos = population[i].MatrizSolucao + val
			# print('crow_pos')
			# print(val)

			# Apply boundary constraints
			for j in range(dimensions[0]):
				crow_pos[j] = np.maximum(crow_pos[j], lower_bounds)
				crow_pos[j] = np.minimum(crow_pos[j], upper_bounds)

			# Evaluate new position
			# crow_fitness = cost_func(crow_pos)
			crow_pos_sol = Solucao(population[0].DadosProblema, deepcopy(crow_pos))
			crow_fitness = crow_pos_sol.fitness

			# Crow behavior
			if crow_fitness < population_fitness[i]:
				# Improvement: update crow position and forget
				population[i].MatrizSolucao = deepcopy(crow_pos)
				population[i].pypsa_update()
				if crow_fitness < best_fitness:
					best_fitness = deepcopy(crow_fitness)
					best_solution = deepcopy(crow_pos)
			else:
				# Exploration: select one of the best crows and follow

				# Select best crows
				# best_crows = population[fitness <= np.percentile(fitness, 100 - pa * 100)]

				# best_crows = population[np.flatnonzero(fitness <= np.percentile(fitness, 100 - pa * 100))]

				# Calculate selection threshold
				selection_threshold = np.percentile(population_fitness, 100 - pa * 100)
				# Select best crows
				best_crows = [individual for individual, fit in zip(population, population_fitness) if fit <= selection_threshold]

				selected_crow = best_crows[np.random.choice(len(best_crows))]
				val2 = np.round(pc * (selected_crow.MatrizSolucao - population[i].MatrizSolucao))
				population[i].MatrizSolucao = population[i].MatrizSolucao + val2
				# print('selected_crow')
				# print(val2)

				# Apply boundary constraints
				for j in range(dimensions[0]):
					crow_pos[j] = np.maximum(crow_pos[j], lower_bounds)
					crow_pos[j] = np.minimum(crow_pos[j], upper_bounds)

				# Evaluate new position
				population_fitness[i] = deepcopy(population[i].fitness)

				if population_fitness[i] < best_fitness:
					best_fitness = deepcopy(population_fitness[i])
					best_solution = deepcopy(population[i])

		print(f'Iteration {iter_num} ')
		print(f'Best Fitness {best_fitness:.2f} ')
		print(population_fitness)
		resultados.append([iter_num, best_fitness])

	print(f'\nTime Elapsed: {(time() - start_time):.2f}s')
	return best_solution, population, resultados


# # Solucao
# f1_bounds = (-100, 100)
# for i in range(10):
# 	bestIndividual, bestFitness = csa(bf.sphere, f1_bounds)
# 	# print("Best individual:", bestIndividual)
# 	print(f"Best fitness: {bestFitness:.4f}")
