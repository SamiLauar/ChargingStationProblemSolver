import numpy as np
from ProblemClass import Problema, Solucao
import SolverFunctions as solver
from typing import List
from copy import deepcopy
from random import random, randint, uniform
from time import time


# Define fitness function
def sphere(x):
	return np.sum(x ** 2)


def differential_evolution(fitness_func, bounds, pop_size=50, dimensions=10, max_generations=200,
                           crossover_prob=0.7, scale_factor=0.7):
	"""
	Implements the Differential Evolution algorithm for optimizing a given fitness function.

	Parameters:
	fitness_func (function): The fitness function to be optimized.
	bounds (array_like): A tuple of arrays specifying the lower and upper bounds for each dimension.
	pop_size (int): The number of individuals in the population. Default is 100.
	dimensions (int): The number of dimensions in the search space. Default is 10.
	crossover_prob (float): The probability of crossover between the target vector and the trial vector. Default is 0.7.
	scale_factor (float): The scale factor used for perturbing the population. Default is 0.7.
	max_generations (int): The maximum number of generations to run the algorithm. Default is 1000.
	min_fitness (float): The minimum fitness value at which to stop the algorithm. Default is 1e-10.
	random_seed (int): The random seed for the random number generator. Default is None.

	Returns:
	The best individual and fitness found by the algorithm.
	"""
	start_time = time()

	# Initialize population
	population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dimensions))

	# Iterate over generations
	for generation in range(max_generations):
		# Iterate over individuals
		for i in range(pop_size):
			# Choose three random vectors from the population
			a, b, c = np.random.choice(pop_size, size=3, replace=False)

			# Generate a trial vector by perturbing the chosen vectors
			trial = population[a] + scale_factor * (population[b] - population[c])

			# Crossover trial vector with individual i
			crossover_mask = np.random.uniform(size=dimensions) < crossover_prob
			offspring = np.where(crossover_mask, trial, population[i])

			# Ensure trial vector has integer values within bounds
			offspring = np.clip(offspring, bounds[0], bounds[1]).astype(float)

			# Evaluate fitness of offspring
			offspring_fitness = fitness_func(offspring)

			# Replace individual i with offspring if fitness is better
			if offspring_fitness < fitness_func(population[i]):
				population[i] = offspring

	# Return best individual and fitness
	best_individual = population[np.argmin([fitness_func(x) for x in population])]
	best_fit = fitness_func(best_individual)
	timeElapsed = time() - start_time
	print(f'Best Fitness: {best_fit}')
	print(f'Time Elapsed: {timeElapsed:.2f} s')
	return best_individual, best_fit, timeElapsed


def DEmod(fitness_func, bounds, pop_size=50, dimensions=10,
          crossover_prob=0.7, scale_factor=0.7, max_generations=200,
          min_fitness=0, random_seed=None):
	"""
	Implements the Differential Evolution algorithm for optimizing a given fitness function.

	Parameters:
	fitness_func (function): The fitness function to be optimized.
	bounds (array_like): A tuple of arrays specifying the lower and upper bounds for each dimension.
	pop_size (int): The number of individuals in the population. Default is 100.
	dimensions (int): The number of dimensions in the search space. Default is 10.
	crossover_prob (float): The probability of crossover between the target vector and the trial vector. Default is 0.7.
	scale_factor (float): The scale factor used for perturbing the population. Default is 0.7.
	max_generations (int): The maximum number of generations to run the algorithm. Default is 1000.
	min_fitness (float): The minimum fitness value at which to stop the algorithm. Default is 1e-10.
	random_seed (int): The random seed for the random number generator. Default is None.

	Returns:
	The best individual and fitness found by the algorithm.
	"""
	start_time = time()
	# Set random seed
	if random_seed is not None:
		np.random.seed(random_seed)

	# Initialize population
	population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dimensions))

	fitness = [fitness_func(x) for x in population]
	best_fit = np.min(fitness)
	best_ind = population[np.argmin(fitness)]

	# Iterate over generations
	for generation in range(max_generations):
		# Iterate over individuals
		for i in range(pop_size):
			# Choose three random vectors from the population
			a, b, c = np.random.choice(pop_size, size=3, replace=False)

			# Crossover trial vector with individual i
			crossover_mask = np.random.uniform(size=dimensions) < crossover_prob

			trial = best_ind + scale_factor * (population[b] - population[c])

			offspring = np.where(crossover_mask, trial, population[i])

			# Evaluate fitness of offspring
			offspring_fitness = fitness_func(offspring)

			# Replace individual i with offspring if fitness is better
			if offspring_fitness < fitness[i]:
				population[i] = deepcopy(offspring)
				fitness[i] = offspring_fitness

				if offspring_fitness < best_fit:
					best_fit = fitness[i]
					best_ind = population[i]

		# Choose three random vectors from the population
		a, b, c = np.random.choice(pop_size, size=3, replace=False)
		# Generate a trial vector by perturbing the chosen vectors
		trialA = population[a] + scale_factor * (population[b] - population[c])
		trialB = population[a] + scale_factor * (population[c] - population[b])
		crossover_mask = np.random.uniform(size=dimensions) < 0.5
		population[np.argmax(fitness)] = np.where(crossover_mask, trialA, trialB)

		# Check for termination condition
		best_fit = np.min([fitness_func(x) for x in population])
		if best_fit < min_fitness:
			break

	# Return best individual and fitness
	best_ind = population[np.argmin([fitness_func(x) for x in population])]
	best_fit = fitness_func(best_ind)
	print(f'Time Elapsed: {(time() - start_time):.2f}s')
	return best_ind, best_fit


def solve(dados_problema, pop_size, bounds, max_generations=200, max_tempo=1000, crossover_prob=0.7, scale_factor=0.7):
	"""
	Implements the Differential Evolution algorithm for optimizing a given fitness function.
	Ensures integer values in the population.

	Parameters:
	fitness_func (function): The fitness function to be optimized.
	bounds (array_like): A tuple of arrays specifying the lower and upper bounds for each dimension.
	pop_size (int): The number of individuals in the population. Default is 100.
	dimensions (int): The number of dimensions in the search space. Default is 10.
	crossover_prob (float): The probability of crossover between the target vector and the trial vector. Default is 0.7.
	scale_factor (float): The scale factor used for perturbing the population. Default is 0.7.
	max_generations (int): The maximum number of generations to run the algorithm. Default is 1000.
	min_fitness (float): The minimum fitness value at which to stop the algorithm. Default is 1e-10.
	random_seed (int): The random seed for the random number generator. Default is None.

	Returns:
	The best individual and fitness found by the algorithm.
	"""

	print('\nDifferential Evolution Start\n')
	start_time = time()

	# Passo 1: Inicialização
	population = list()  # Equivale a População
	for i in range(pop_size):
		population.append(solver.gerarSolucaoAleatoriaZonas(dados_problema))

	dimensions = population[0].MatrizSolucao.shape
	population_fitness = np.array([sol.fitness for sol in population])
	best_fitness = np.min(population_fitness)
	resultados = [[0, best_fitness]]

	print(f'Generation  \033[36m0\033[m  -  Best Fitness = \033[31m{best_fitness:.3f}\033[m  -  Time: \033[33m{(time() - start_time):^6.1f}\033[m s')

	resetou = False
	# Iterate over generations
	for generation in range(1, max_generations + 1):
		# Iterate over individuals

		# if tick >= 0.1*max_generations:
		if ((generation > 0.5*max_generations) or ((time() - start_time) > 0.5*max_tempo)) and not resetou:
			resetou = True
			print('Population Reset')
			population[0] = deepcopy(population[np.argmin(population_fitness)])
			for i in range(1, pop_size):
				population[i] = solver.gerarSolucaoAleatoriaZonas(dados_problema)
			population_fitness = np.array([sol.fitness for sol in population])

		for i in range(pop_size):

			# Choose three random vectors from the population
			a, b, c = np.random.choice(pop_size, size=3, replace=False)

			# Generate a trial vector by perturbing the chosen vectors
			trial = population[a].MatrizSolucao + scale_factor * (
					population[b].MatrizSolucao - population[c].MatrizSolucao)

			# Ensure trial vector has integer values within bounds
			trial = np.clip(trial, bounds[0], bounds[1]).astype(int)

			# Crossover trial vector with individual i
			crossover_mask = np.random.uniform(size=dimensions) < crossover_prob
			offspring = np.where(crossover_mask, trial, population[i].MatrizSolucao)

			# Ensure offspring has integer values within bounds
			offspring = np.clip(offspring, bounds[0], bounds[1]).astype(int)

			# Evaluate fitness of offspring
			# offspring_fitness = solver.FitnessCalculate(offspring, population[0].DadosProblema)
			offspring_sol = Solucao(dados_problema, deepcopy(offspring))

			# Replace individual i with offspring if fitness is better
			'''
			if (offspring_sol.fitnessA < population[i].fitnessA) and (offspring_sol.fitnessB > population[i].fitnessB):
				print(' !', end='')
				population[i].MatrizSolucao = deepcopy(offspring_sol.MatrizSolucao)
				population[i].pypsa_update()
				population_fitness[i] = population[i].fitness
			'''
			if offspring_sol.fitness < population_fitness[i]:
				population[i].MatrizSolucao = deepcopy(offspring_sol.MatrizSolucao)
				population[i].pypsa_update()
				population_fitness[i] = population[i].fitness
				if population_fitness[i] < best_fitness:
					best_fitness = population_fitness[i]

		best_fitness = np.min(population_fitness)
		print(f'Generation \033[36m{generation:^3}\033[m -  Best Fitness = \033[31m{best_fitness:.3f}\033[m  -  Time: \033[33m{(time() - start_time):^6.1f}\033[m s')
		# print(population_fitness)

		resultados.append([generation, best_fitness])

		if time() - start_time > max_tempo:
			break

	# print(f'Best Fitness = \033[31m{best_fitness:.3f}\033[m ')
	# Return best individual
	best_individual = population[np.argmin(population_fitness)]
	# best_individual = solver.ParetoDominante(population)
	return best_individual, population, resultados


def hybridMVO(dados_problema, pop_size, bounds, max_generations=200, max_tempo=1000, crossover_prob=0.7, scale_factor=0.7):
	"""
	Implements the Differential Evolution algorithm for optimizing a given fitness function.
	Ensures integer values in the population.

	Parameters:
	fitness_func (function): The fitness function to be optimized.
	bounds (array_like): A tuple of arrays specifying the lower and upper bounds for each dimension.
	pop_size (int): The number of individuals in the population. Default is 100.
	dimensions (int): The number of dimensions in the search space. Default is 10.
	crossover_prob (float): The probability of crossover between the target vector and the trial vector. Default is 0.7.
	scale_factor (float): The scale factor used for perturbing the population. Default is 0.7.
	max_generations (int): The maximum number of generations to run the algorithm. Default is 1000.
	min_fitness (float): The minimum fitness value at which to stop the algorithm. Default is 1e-10.
	random_seed (int): The random seed for the random number generator. Default is None.

	Returns:
	The best individual and fitness found by the algorithm.
	"""

	print('\nDE-MVO Start\n')
	start_time = time()

	# Passo 1: Inicialização
	population = list()  # Equivale a População
	for i in range(pop_size):
		population.append(solver.gerarSolucaoAleatoriaZonas(dados_problema))

	dimensions = population[0].MatrizSolucao.shape
	population_fitness = np.array([sol.fitness for sol in population])
	best_individual = population[np.argmin(population_fitness)]
	best_fitness = np.min(population_fitness)
	resultados = [[0, best_fitness]]

	print(f'Generation  \033[36m0\033[m  -  Best Fitness = \033[31m{best_fitness:.3f}\033[m')
	resetou = False

	# Iterate over generations
	for generation in range(1, max_generations + 1):

		if ((generation > 0.5*max_generations) or ((time() - start_time) > 0.5*max_tempo)) and not resetou:
			resetou = True
			print('Population Reset')
			population[0] = deepcopy(population[np.argmin(population_fitness)])
			for i in range(1, pop_size):
				population[i] = solver.gerarSolucaoAleatoriaZonas(dados_problema)
			population_fitness = np.array([sol.fitness for sol in population])

		# Wormhole_existance_probability e Travelling Distance Rate
		WEP, TDR = update_WEP_TDR(generation, max_generations)
		fit_pu = np.max(population_fitness)

		# Iterate over individuals
		for i in range(pop_size):

			# Choose three random vectors from the population
			a, b, c = np.random.choice(pop_size, size=3, replace=False)

			# Generate a trial vector by perturbing the chosen vectors + WormHole
			if np.random.random() < WEP:
				distance_travel = TDR * bounds[1] * random()
				r_mask = np.random.uniform(size=dimensions) < 0.5
				WormHole_Universe = np.where(r_mask,
				                             best_individual.MatrizSolucao + distance_travel,
				                             best_individual.MatrizSolucao - distance_travel)

				trial = best_individual.MatrizSolucao + scale_factor * (population[b].MatrizSolucao - population[c].MatrizSolucao) + WormHole_Universe
				# trial = population[a].MatrizSolucao + scale_factor * (
				# 		population[b].MatrizSolucao - population[c].MatrizSolucao) + WormHole_Universe
			else:
				trial = best_individual.MatrizSolucao + scale_factor * (population[b].MatrizSolucao - population[c].MatrizSolucao)
				# trial = population[a].MatrizSolucao + scale_factor * (
				# 		population[b].MatrizSolucao - population[c].MatrizSolucao)

			# Ensure trial vector has integer values within bounds
			trial = np.clip(trial, bounds[0], bounds[1]).astype(int)

			# Crossover trial vector with individual i
			crossover_mask = np.random.uniform(size=dimensions) < crossover_prob
			offspring = np.where(crossover_mask, trial, population[i].MatrizSolucao)

			# Ensure offspring has integer values within bounds
			offspring = np.clip(offspring, bounds[0], bounds[1]).astype(int)

			# Evaluate fitness of offspring
			offspring_sol = Solucao(dados_problema, deepcopy(offspring))

			# Replace individual i with offspring if fitness is better
			if offspring_sol.fitness < population[i].fitness:
				population[i].MatrizSolucao = deepcopy(offspring_sol.MatrizSolucao)
				population[i].pypsa_update()
				population_fitness[i] = population[i].fitness
				if population_fitness[i] < best_fitness:
					best_individual = population[i]
					best_fitness = population_fitness[i]

			# Passo 2: Aparição do White Hole: Inflation Rate Roulette
			if random() < population_fitness[i]/fit_pu:

				White_hole_index = roulette_wheel_selection(weights=-population_fitness)

				crossover_mask = np.random.uniform(size=dimensions) < population_fitness[i]/fit_pu
				BWHole_Universe = np.where(crossover_mask, population[i].MatrizSolucao,
				                           population[White_hole_index].MatrizSolucao)

				# Ensure offspring has integer values within bounds
				BWHole_Universe = np.clip(BWHole_Universe, bounds[0], bounds[1]).astype(int)
				# Evaluate fitness of offspring
				offspring_sol = Solucao(dados_problema, deepcopy(BWHole_Universe))

				# Replace individual i with offspring if fitness is better
				if offspring_sol.fitness < population[i].fitness:
					population[i].MatrizSolucao = deepcopy(offspring_sol.MatrizSolucao)
					population[i].pypsa_update()
					population_fitness[i] = population[i].fitness
					if population_fitness[i] < best_fitness:
						best_individual = population[i]
						best_fitness = population_fitness[i]

		print(f'Generation \033[36m{generation:^3}\033[m -  Best Fitness = \033[31m{best_fitness:.3f}\033[m  - Time: \033[33m{(time() - start_time):^6.1f} s\033[m')
		# print(population_fitness)

		resultados.append([generation, best_fitness])

		if time() - start_time > max_tempo:
			break

	# print(f'best Fitness {best_fitness:.3f} ')
	# Return best individual
	best_individual = population[np.argmin(population_fitness)]
	return best_individual, population, resultados


def hybridMVOA(population: List[Solucao], bounds, max_generations=200, crossover_prob=0.7, scale_factor=0.7):
	"""
	Implements the Differential Evolution algorithm for optimizing a given fitness function.
	Ensures integer values in the population.

	Parameters:
	fitness_func (function): The fitness function to be optimized.
	bounds (array_like): A tuple of arrays specifying the lower and upper bounds for each dimension.
	pop_size (int): The number of individuals in the population. Default is 100.
	dimensions (int): The number of dimensions in the search space. Default is 10.
	crossover_prob (float): The probability of crossover between the target vector and the trial vector. Default is 0.7.
	scale_factor (float): The scale factor used for perturbing the population. Default is 0.7.
	max_generations (int): The maximum number of generations to run the algorithm. Default is 1000.
	min_fitness (float): The minimum fitness value at which to stop the algorithm. Default is 1e-10.
	random_seed (int): The random seed for the random number generator. Default is None.

	Returns:
	The best individual and fitness found by the algorithm.
	"""

	print('\nDifferential Evolution Start\n')

	pop_size = len(population)
	dimensions = population[0].MatrizSolucao.shape
	dados_problema = population[0].DadosProblema

	population_fitness = np.array([sol.fitness for sol in population])
	best_individual = population[np.argmin(population_fitness)]
	best_fitness = np.min(population_fitness)
	resultados = [[0, best_fitness]]

	print(f'start')
	print(f'best Fitness {best_fitness:.3f} ')

	# Iterate over generations
	for generation in range(1, max_generations + 1):

		# Wormhole_existance_probability e Travelling Distance Rate
		WEP, TDR = update_WEP_TDR(generation, max_generations)

		# Iterate over individuals
		for i in range(pop_size):

			# Choose three random vectors from the population
			a, b, c = np.random.choice(pop_size, size=3, replace=False)

			# Generate a trial vector by perturbing the chosen vectors
			trial = best_individual.MatrizSolucao + scale_factor * (
					population[b].MatrizSolucao - population[c].MatrizSolucao)
			# trial = population[a].MatrizSolucao + scale_factor * (
			# 		population[b].MatrizSolucao - population[c].MatrizSolucao)

			# Ensure trial vector has integer values within bounds
			trial = np.clip(trial, bounds[0], bounds[1]).astype(int)

			# Crossover trial vector with individual i
			crossover_mask = np.random.uniform(size=dimensions) < crossover_prob
			offspring = np.where(crossover_mask, trial, population[i].MatrizSolucao)

			# Ensure offspring has integer values within bounds
			offspring = np.clip(offspring, bounds[0], bounds[1]).astype(int)

			# Evaluate fitness of offspring
			# offspring_fitness = solver.FitnessCalculate(offspring, population[0].DadosProblema)
			offspring_sol = Solucao(dados_problema, deepcopy(offspring))

			# Replace individual i with offspring if fitness is better
			if offspring_sol.fitness < population[i].fitness:
				population[i].MatrizSolucao = deepcopy(offspring_sol.MatrizSolucao)
				population[i].pypsa_update()
				population_fitness[i] = population[i].fitness
				if population_fitness[i] < best_fitness:
					best_individual = population[i]
					best_fitness = population_fitness[i]

			# Passo 3: Aparição do Worm Hole
			if random() < WEP:  # Wormhole_existance_probability
				r3_maskA = np.random.uniform(size=dimensions) < 0.5
				r3_maskB = np.random.uniform(size=dimensions) < 0.5
				r4 = random()
				distance_travel = np.round(TDR * bounds[1] * r4)

				WormHole_Universe = np.where(r3_maskA,
				                             best_individual.MatrizSolucao + distance_travel,
				                             best_individual.MatrizSolucao - distance_travel)

				WormHole_Universe = np.where(r3_maskB, population[i].MatrizSolucao, WormHole_Universe)

				# Ensure offspring has integer values within bounds
				WormHole_Universe = np.clip(WormHole_Universe, bounds[0], bounds[1]).astype(int)
				# Evaluate fitness of offspring
				offspring_sol = Solucao(dados_problema, deepcopy(WormHole_Universe))

				# Replace individual i with offspring if fitness is better
				if offspring_sol.fitness < population[i].fitness:
					population[i].MatrizSolucao = deepcopy(offspring_sol.MatrizSolucao)
					population[i].pypsa_update()
					population_fitness[i] = population[i].fitness
					if population_fitness[i] < best_fitness:
						best_individual = population[i]
						best_fitness = population_fitness[i]

		# population_fitness = np.array([sol.fitness for sol in population])
		# best_fitness = np.min(population_fitness)

		print(f'Generation {generation} ')
		print(f'Best Fitness {best_fitness:.3f} ')
		print(population_fitness)

		resultados.append([generation, best_fitness])


	print(f'best Fitness {best_fitness:.3f} ')
	# Return best individual
	best_individual = population[np.argmin(population_fitness)]
	return best_individual, population, resultados


def hybridMVOB(population: List[Solucao], bounds, max_generations=200, crossover_prob=0.7, scale_factor=0.7):
	"""
	Implements the Differential Evolution algorithm for optimizing a given fitness function.
	Ensures integer values in the population.

	Parameters:
	fitness_func (function): The fitness function to be optimized.
	bounds (array_like): A tuple of arrays specifying the lower and upper bounds for each dimension.
	pop_size (int): The number of individuals in the population. Default is 100.
	dimensions (int): The number of dimensions in the search space. Default is 10.
	crossover_prob (float): The probability of crossover between the target vector and the trial vector. Default is 0.7.
	scale_factor (float): The scale factor used for perturbing the population. Default is 0.7.
	max_generations (int): The maximum number of generations to run the algorithm. Default is 1000.
	min_fitness (float): The minimum fitness value at which to stop the algorithm. Default is 1e-10.
	random_seed (int): The random seed for the random number generator. Default is None.

	Returns:
	The best individual and fitness found by the algorithm.
	"""

	print('\nDE-MVO B Start\n')

	pop_size = len(population)
	dimensions = population[0].MatrizSolucao.shape
	dados_problema = population[0].DadosProblema

	population_fitness = np.array([sol.fitness for sol in population])
	best_individual = population[np.argmin(population_fitness)]
	best_fitness = np.min(population_fitness)
	resultados = [[0, best_fitness]]

	print(f'start')
	print(f'best Fitness {best_fitness:.2f} ')
	resetou = False

	# Iterate over generations
	for generation in range(1, max_generations + 1):

		if generation > 0.5*max_generations and not resetou:
			resetou = True
			print('Population Reset')
			population[0] = deepcopy(population[np.argmin(population_fitness)])
			for i in range(1, pop_size):
				population[i] = solver.gerarSolucaoAleatoriaZonas(dados_problema)
			population_fitness = np.array([sol.fitness for sol in population])

		# Wormhole_existance_probability e Travelling Distance Rate
		WEP, TDR = update_WEP_TDR(generation, max_generations)

		# Iterate over individuals
		for i in range(pop_size):

			# Choose three random vectors from the population
			a, b, c = np.random.choice(pop_size, size=3, replace=False)

			distance_travel = TDR * bounds[1] * random()
			r_mask = np.random.uniform(size=dimensions) < 0.5
			WormHole_Universe = np.where(r_mask,
			                             best_individual.MatrizSolucao + distance_travel,
			                             best_individual.MatrizSolucao - distance_travel)

			# Generate a trial vector by perturbing the chosen vectors
			# trial = best_individual.MatrizSolucao + scale_factor * (
			# 		population[b].MatrizSolucao - population[c].MatrizSolucao)
			trial = population[a].MatrizSolucao + scale_factor * (
					population[b].MatrizSolucao - population[c].MatrizSolucao) + WormHole_Universe

			# Ensure trial vector has integer values within bounds
			trial = np.clip(trial, bounds[0], bounds[1]).astype(int)

			# Crossover trial vector with individual i
			crossover_mask = np.random.uniform(size=dimensions) < crossover_prob
			offspring = np.where(crossover_mask, trial, population[i].MatrizSolucao)

			# Ensure offspring has integer values within bounds
			offspring = np.clip(offspring, bounds[0], bounds[1]).astype(int)

			# Evaluate fitness of offspring
			# offspring_fitness = solver.FitnessCalculate(offspring, population[0].DadosProblema)
			offspring_sol = Solucao(dados_problema, deepcopy(offspring))

			# Replace individual i with offspring if fitness is better
			if offspring_sol.fitness < population[i].fitness:
				population[i].MatrizSolucao = deepcopy(offspring_sol.MatrizSolucao)
				population[i].pypsa_update()
				population_fitness[i] = population[i].fitness
				if population_fitness[i] < best_fitness:
					best_individual = population[i]
					best_fitness = population_fitness[i]

		print(f'Generation {generation} ')
		print(f'Best Fitness {best_fitness:.2f} ')
		print(population_fitness)

		resultados.append([generation, best_fitness])

	print(f'best Fitness {best_fitness:.2f} ')
	# Return best individual
	best_individual = population[np.argmin(population_fitness)]
	return best_individual, population, resultados


def DE_MVO(fitness_func, bounds, pop_size=50, dimensions=10, max_generations=200, crossover_prob=0.7, scale_factor=0.7, min_fitness=0):
	"""
	Implements the Differential Evolution algorithm for optimizing a given fitness function.
	Ensures integer values in the population.

	Parameters:
	fitness_func (function): The fitness function to be optimized.
	bounds (array_like): A tuple of arrays specifying the lower and upper bounds for each dimension.
	pop_size (int): The number of individuals in the population. Default is 100.
	dimensions (int): The number of dimensions in the search space. Default is 10.
	crossover_prob (float): The probability of crossover between the target vector and the trial vector. Default is 0.7.
	scale_factor (float): The scale factor used for perturbing the population. Default is 0.7.
	max_generations (int): The maximum number of generations to run the algorithm. Default is 1000.
	min_fitness (float): The minimum fitness value at which to stop the algorithm. Default is 1e-10.
	random_seed (int): The random seed for the random number generator. Default is None.

	Returns:
	The best individual and fitness found by the algorithm.
	"""

	start_time = time()

	# Initialize population
	population = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dimensions))

	population_fitness = np.array([fitness_func(sol) for sol in population])
	best_individual = population[np.argmin(population_fitness)]
	best_fitness = np.min(population_fitness)
	resetou = False

	# Iterate over generations
	for generation in range(max_generations):

		if generation > 0.5*max_generations and not resetou:
			resetou = True
			population[0] = deepcopy(population[np.argmin(population_fitness)])
			for i in range(1, pop_size):
				population[i] = np.random.uniform(bounds[0], bounds[1], size=dimensions)
			population_fitness = np.array([fitness_func(sol) for sol in population])

		# Wormhole_existance_probability e Travelling Distance Rate
		WEP, TDR = update_WEP_TDR(generation, max_generations)
		fit_pu = np.max(population_fitness)

		# Iterate over individuals
		for i in range(pop_size):

			# Choose three random vectors from the population
			a, b, c = np.random.choice(pop_size, size=3, replace=False)

			# Generate a trial vector by perturbing the chosen vectors + WormHole
			if np.random.random() < WEP:
				# distance_travel = TDR * bounds[1] * random()
				distance_travel = TDR * (bounds[1] - bounds[0]) * random() + bounds[0]
				r_mask = np.random.uniform(size=dimensions) < 0.5
				WormHole_Universe = np.where(r_mask,
				                             best_individual + distance_travel,
				                             best_individual - distance_travel)

				trial = best_individual + scale_factor * (population[b] - population[c]) + WormHole_Universe
			else:
				trial = best_individual + scale_factor * (population[b] - population[c])

			# Crossover trial vector with individual i
			crossover_mask = np.random.uniform(size=dimensions) < crossover_prob
			offspring = np.where(crossover_mask, trial, population[i])

			# Ensure offspring has integer values within bounds
			offspring = np.clip(offspring, bounds[0], bounds[1]).astype(float)
			# Evaluate fitness of offspring
			offspring_fitness = fitness_func(offspring)

			# Replace individual i with offspring if fitness is better
			if offspring_fitness < population_fitness[i]:
				population[i] = offspring
				population_fitness[i] = offspring_fitness
				if offspring_fitness < best_fitness:
					best_individual = population[i]
					best_fitness = offspring_fitness

			# Passo 2: Aparição do White Hole: Inflation Rate Roulette
			if random() < population_fitness[i]/fit_pu:

				White_hole_index = roulette_wheel_selection(weights=[-fit/fit_pu for fit in population_fitness])

				crossover_mask = np.random.uniform(size=dimensions) < population_fitness[i]/fit_pu
				BWHole_Universe = np.where(crossover_mask, population[i],
				                           population[White_hole_index])

				# Ensure offspring has integer values within bounds
				BWHole_Universe = np.clip(BWHole_Universe, bounds[0], bounds[1]).astype(float)
				# Evaluate fitness of offspring
				offspring_fitness = fitness_func(BWHole_Universe)

				# Replace individual i with offspring if fitness is better
				if offspring_fitness < population_fitness[i]:
					population[i] = BWHole_Universe
					population_fitness[i] = offspring_fitness
					if offspring_fitness < best_fitness:
						best_individual = population[i]
						best_fitness = offspring_fitness
		if best_fitness <= min_fitness:
			break

	best_individual = population[np.argmin(population_fitness)]
	print(f'best Fitness {best_fitness:.3f} ')
	# Return best individual
	timeElapsed = time() - start_time
	print(f'Time Elapsed: {timeElapsed:.2f}s')
	return best_individual, best_fitness, timeElapsed


# WEP: Wormhole Existence Probability
# TDR: Travelling Distance Rate
def update_WEP_TDR(n_iter: int, max_iter: int, min_wep: float = 0.2, max_wep: float = 1, min_tdr=0.15, p: float = 0.6):
	wep = min_wep + n_iter * (max_wep - min_wep) / max_iter
	tdr = 1 - (1 - min_tdr) * (n_iter / max_iter) ** (1 / p)

	return wep, tdr


def roulette_wheel_selection(weights):
	accumulation = np.cumsum(weights)
	p = np.random.rand() * accumulation[-1]
	chosen_index = -1
	for index in range(len(accumulation)):
		if accumulation[index] < p:
			chosen_index = index
			break
	return chosen_index

# # Solucao
# f1_bounds = (-100, 100)
# for i in range(10):
# 	bestIndividual, bestFitness = differential_evolution(sphere, f1_bounds)
# 	# bestIndividual, bestFitness = DEmod(sphere, f1_bounds)
# 	print("Best individual:", bestIndividual)
# 	print("Best fitness:", bestFitness)
