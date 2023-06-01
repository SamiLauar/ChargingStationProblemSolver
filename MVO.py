from copy import deepcopy
import pandas as pd
import numpy as np
from ProblemClass import Problema, Solucao
import SolverFunctions as solver
from typing import List
from random import random, randint, uniform
from time import time
import BenchmarkFuntions as bf


def mvo(fitness_func, bounds, pop_size=50, dimensions=10, max_iter=200):
	start_time = time()

	# Initialize population
	universes = np.random.uniform(bounds[0], bounds[1], size=(pop_size, dimensions))

	# Utilidades e Vetor de Resultados por iteração
	universes_fitness = np.array([fitness_func(x) for x in universes])
	best_fitness = np.min(universes_fitness)
	best_universe = universes[np.argmin(universes_fitness)]

	# Loop Solucao
	for iter_num in range(max_iter):

		fobj_pu = np.max(universes_fitness)
		Fitness_pu = np.array([fit / fobj_pu for fit in universes_fitness])
		# Wormhole_existance_probability e Travelling Distance Rate
		WEP, TDR = update_WEP_TDR(iter_num, max_iter)

		for i in range(pop_size):
			# Passo 2: Aparição do White Hole: Inflation Rate Roulette
			if random() < Fitness_pu[i]:

				White_hole_index = roulette_wheel_selection(weights=-1 * Fitness_pu)
				crossover_mask = np.random.uniform(size=dimensions) < Fitness_pu[i]
				BWHole_Universe = np.where(crossover_mask, universes[i], universes[White_hole_index])

				# Ensure offspring has integer values within bounds
				BWHole_Universe = np.clip(BWHole_Universe, bounds[0], bounds[1]).astype(float)
				# Evaluate fitness of offspring
				offspring_fitness = fitness_func(BWHole_Universe)

				# Replace individual i with offspring if fitness is better
				if offspring_fitness < universes_fitness[i]:
					universes[i] = BWHole_Universe
					universes_fitness[i] = offspring_fitness
					if offspring_fitness < best_fitness:
						best_universe = universes[i]
						best_fitness = offspring_fitness

			# Passo 3: Aparição do Worm Hole
			if random() < WEP:  # Wormhole_existance_probability
				r3_maskA = np.random.uniform(size=dimensions) < 0.5
				r3_maskB = np.random.uniform(size=dimensions) < 0.5
				r4 = random()
				distance_travel = TDR * (bounds[1] - bounds[0]) * r4 + bounds[0]

				WormHole_Universe = np.where(r3_maskA,
				                             best_universe + distance_travel,
				                             best_universe - distance_travel)

				WormHole_Universe = np.where(r3_maskB, universes[i], WormHole_Universe)

				# Ensure offspring has integer values within bounds
				WormHole_Universe = np.clip(WormHole_Universe, bounds[0], bounds[1]).astype(float)
				# Evaluate fitness of offspring
				offspring_fitness = fitness_func(WormHole_Universe)

				# Replace individual i with offspring if fitness is better
				if offspring_fitness < universes_fitness[i]:
					universes[i] = WormHole_Universe
					universes_fitness[i] = offspring_fitness
					if offspring_fitness < best_fitness:
						best_universe = universes[i]
						best_fitness = offspring_fitness

		universes_fitness = np.array([fitness_func(x) for x in universes])
		best_fitness = np.min(universes_fitness)

	timeElapsed = time() - start_time
	# print(f'Time Elapsed: {timeElapsed:.2f}s')

	best_universe = universes[np.argmin(universes_fitness)]
	return best_universe, best_fitness, timeElapsed


def solveOriginal(universes: List[Solucao], max_iter: int, max_X):
	in_solve = time()

	t_BestUniverseIndex = list()
	t_WhiteHole = list()
	t_WormHole = list()

	# Variaveis
	num_universos = len(universes)
	tamSolucao = len(universes[0].MatrizSolucao)

	# Utilidades e Vetor de Resultados por iteração
	bestUnIndex = solver.BestUniverseIndex(universes, 0)
	universes_fitness = np.array([sol.fitness for sol in universes])
	bestFitInicial = universes[bestUnIndex].fitness
	resultados = [[0, bestFitInicial]]

	print(f'start')
	print(f'best Fitness {bestFitInicial:.2f} ')

	population_fitness = np.array([sol.fitness for sol in universes])
	# Loop Solucao
	for iter_num in range(1, max_iter + 1):

		fit_pu = np.max(universes_fitness)
		un_fitness_pu = np.array([fit / fit_pu for fit in universes_fitness])

		for i in range(num_universos):
			# Wormhole_existance_probability e Travelling Distance Rate
			WEP, TDR = update_WEP_TDR(iter_num, max_iter, max_X)

			# Checa se o atual universo é o melhor, pois se for, ele não sofrerá alterações que piorem a sua Fitness_pu
			in_BestUniverseIndex = time()
			bestUnIndex = solver.BestUniverseIndex(universes, bestUnIndex)
			t_BestUniverseIndex.append(time() - in_BestUniverseIndex)
			if bestUnIndex == i:
				bestUn = True
			else:
				bestUn = False

			for j in range(tamSolucao):
				# Passo 2: Aparição do White Hole: Inflation Rate Roulette
				r1 = random()
				if r1 < population_fitness[i]/np.max(population_fitness):
					in_WhiteHole = time()
					createWhiteHole(universes, i, j, un_fitness_pu)
					t_WhiteHole.append(time() - in_WhiteHole)

				r2 = random()
				if r2 < WEP:  # Wormhole_existance_probability
					r3 = random()
					if r3 < 0.5:
						in_WormHole = time()
						# Troca o eletroposto de lugar ou muda sua potencia? pode fazer += pot_cs ou -= pot_cs
						createWormHole(universes, TDR, i, j, un_fitness_pu, bestUnIndex)
						t_WormHole.append(time() - in_WormHole)

		population_fitness = np.array([sol.fitness for sol in universes])
		best_fitness = np.min(population_fitness)

		print(f'Iter: {iter_num} ')
		print(f'Best Fitness: {best_fitness:.2f} ')
		print(population_fitness)
		# solver.ObjMediaPrint(population, np.argmin(population_fitness))

		resultados.append([iter_num, best_fitness])

	print(f't_WormHole: {sum(t_WormHole):.2f}s in {len(t_WormHole)}')
	print(f't_WhiteHole: {sum(t_WhiteHole):.2f}s in {len(t_WhiteHole)}')
	print(f'T tot BestUniverseIndex: {sum(t_BestUniverseIndex):.2f}s')
	print(f'T solve: {(time() - in_solve):.2f}s')

	best_universe = solver.BestUniverse(universes, bestUnIndex)
	return best_universe, universes, resultados


def solve(dados_problema, num_universos, bounds, max_iter: int, max_tempo):
	start_time = time()

	# Passo 1: Inicialização
	universes = list()  # Equivale a População
	for i in range(num_universos):
		universes.append(solver.gerarSolucaoAleatoriaZonas(dados_problema))

	# Variaveis
	dimensions = universes[0].MatrizSolucao.shape
	universes_fitness = np.array([sol.fitness for sol in universes])
	best_universe = universes[np.argmin(universes_fitness)]
	best_fitness = np.min(universes_fitness)
	# Utilidades e Vetor de Resultados por iteração
	resultados = [[0, best_fitness]]
	resetou = False

	print(f'MVO Start')
	print(f'Iteration  \033[36m0\033[m  -  Best Fitness = \033[31m{best_fitness:.3f}\033[m')

	# Loop Solucao
	for iter_num in range(1, max_iter + 1):

		if ((iter_num > 0.5*max_iter) or ((time() - start_time) > 0.5*max_tempo)) and not resetou:
			resetou = True
			print('Population Reset')
			universes[0] = deepcopy(universes[np.argmin(universes_fitness)])
			for i in range(1, num_universos):
				universes[i] = solver.gerarSolucaoAleatoriaZonas(dados_problema)
			universes_fitness = np.array([sol.fitness for sol in universes])

		fit_pu = np.max(universes_fitness)
		un_fitness_pu = universes_fitness/fit_pu

		# Wormhole_existance_probability e Travelling Distance Rate
		WEP, TDR = update_WEP_TDR(iter_num, max_iter)

		for i in range(num_universos):
			# universes_fitness = np.array([sol.fitness for sol in universes])
			# best_universe = universes[np.argmin(universes_fitness)]

			# Passo 2: Aparição do White Hole: Inflation Rate Roulette
			if random() < un_fitness_pu[i]:  # Normalized_Fitness = NI[i]:
				# Nao Percebi muita diferença de usar [ 1/fit ] ou [ -fit ] ou [ fit ]
				White_hole_index = roulette_wheel_selection(weights=-un_fitness_pu)

				crossover_mask = np.random.uniform(size=dimensions) < un_fitness_pu[i]
				BWHole_Universe = np.where(crossover_mask, universes[i].MatrizSolucao,
				                           universes[White_hole_index].MatrizSolucao)

				# Ensure offspring has integer values within bounds
				BWHole_Universe = np.clip(BWHole_Universe, bounds[0], bounds[1]).astype(int)
				# Evaluate fitness of offspring
				offspring_sol = Solucao(dados_problema, deepcopy(BWHole_Universe))

				if offspring_sol.fitness < universes_fitness[i]:
					universes[i].MatrizSolucao = deepcopy(offspring_sol.MatrizSolucao)
					universes[i].pypsa_update()
					universes_fitness[i] = universes[i].fitness
					un_fitness_pu[i] = universes_fitness[i]/fit_pu
					if universes_fitness[i] < best_fitness:
						best_universe = universes[i]
						best_fitness = universes_fitness[i]

			# Passo 3: Aparição do Worm Hole
			if random() < WEP:  # Wormhole_existance_probability
				r3_maskA = np.random.uniform(size=dimensions) < 0.5
				r3_maskB = np.random.uniform(size=dimensions) < 0.5
				r4 = random()
				distance_travel = np.round(TDR * bounds[1] * r4)

				WormHole_Universe = np.where(r3_maskA,
				                             best_universe.MatrizSolucao + distance_travel,
				                             best_universe.MatrizSolucao - distance_travel)

				WormHole_Universe = np.where(r3_maskB, universes[i].MatrizSolucao, WormHole_Universe)

				# Ensure offspring has integer values within bounds
				WormHole_Universe = np.clip(WormHole_Universe, bounds[0], bounds[1]).astype(int)
				# Evaluate fitness of offspring
				offspring_sol = Solucao(dados_problema, deepcopy(WormHole_Universe))

				if offspring_sol.fitness < universes_fitness[i]:
					universes[i].MatrizSolucao = deepcopy(offspring_sol.MatrizSolucao)
					universes[i].pypsa_update()
					universes_fitness[i] = universes[i].fitness
					un_fitness_pu[i] = universes_fitness[i] / fit_pu
					if universes_fitness[i] < best_fitness:
						best_universe = universes[i]
						best_fitness = universes_fitness[i]

		# universes_fitness = np.array([sol.fitness for sol in universes])
		# best_fitness = np.min(universes_fitness)
		print(f'Generation \033[36m{iter_num:^3}\033[m -  Best Fitness = \033[31m{best_fitness:.3f}\033[m  - Time: \033[33m{(time() - start_time):^6.1f}\033[m s')
		# print(universes_fitness)
		resultados.append([iter_num, best_fitness])
		if time() - start_time > max_tempo:
			break

	# print(f'Time Elapsed: {(time() - start_time):.3f}s')
	best_universe = universes[np.argmin(universes_fitness)]
	return best_universe, universes, resultados


# WEP: Wormhole Existence Probability
# TDR: Travelling Distance Rate
def update_WEP_TDR(n_iter: int, max_iter: int, min_wep: float = 0.4, max_wep: float = 1, min_tdr: float = 0.15,
                   p: float = 0.6):
	wep = min_wep + n_iter * (max_wep - min_wep) / max_iter
	tdr = 1 - (1 - min_tdr) * (n_iter / max_iter) ** (1 / p)
	return wep, tdr


# Entrada: weights = 1./ranks
# ranks(i)=ranks(i)+1;
# apenas sobe a pontuação "rank" se o universo i for melhor em todas as func obj que o universo j,
# mudar para 1 ponto por função?
def roulette_wheel_selection(weights):
	accumulation = np.cumsum(weights)
	p = np.random.rand() * accumulation[-1]
	chosen_index = -1
	for index in range(len(accumulation)):
		if accumulation[index] < p:
			chosen_index = index
			break
	return chosen_index


def createWhiteHole(universes: List[Solucao], i: int, j: int, un_fitness_pu):
	fit_antigo = un_fitness_pu[i]
	Solucao_copy = [pot for pot in universes[i].MatrizSolucao[j]]

	# Nao Percebi muita diferença de usar [ 1/fit ] ou [ -fit ] ou [ fit ]
	White_hole_index = roulette_wheel_selection(weights=[-fit_pu for fit_pu in un_fitness_pu])

	universes[i].MatrizSolucao[j] = universes[White_hole_index].MatrizSolucao[j]

	# Depois de qualquer alteração na matriz universos, é necesário recalcular o fitness e o power flow
	# solver.FitnessUpdate(universes, i)
	universes[i].pypsa_update()

	if not universes[i].factivel or (un_fitness_pu[i] > fit_antigo):
		universes[i].MatrizSolucao[j] = deepcopy(Solucao_copy)
		un_fitness_pu[i] = deepcopy(fit_antigo)


def createWormHole(universes: List[Solucao], TDR, i: int, j: int, un_fitness_pu, bestUnIndex: int):
	pot_cs = universes[i].DadosProblema.PotEletroposto

	fit_antigo = un_fitness_pu[i]
	pot_antigo = [pot for pot in universes[i].MatrizSolucao[j]]

	# r4 = random()
	# [maior_j, menor_j] = solver.Bounds(universes, j)
	maior_j = universes[0].DadosProblema.MaximoEletropostos
	best_universe = solver.BestUniverse(universes, bestUnIndex)

	for k in range(len(pot_cs)):
		# Criar função de modificar usando TDR ainda mantendo a função factivel? É possivel?
		universes[i].MatrizSolucao[j, k] = best_universe.MatrizSolucao[j, k] - round(TDR * maior_j)
		if universes[i].MatrizSolucao[j, k] < 0:
			universes[i].MatrizSolucao[j, k] *= -1

	# Depois de qualquer alteração na matriz universos, é necesário recalcular o fitness e o power flow
	# solver.FitnessUpdate(universes, i)
	universes[i].pypsa_update()

	if not universes[i].factivel or (un_fitness_pu[i] > fit_antigo):
		universes[i].MatrizSolucao[j] = deepcopy(pot_antigo)
		un_fitness_pu[i] = deepcopy(fit_antigo)


# # Solucao
# f1_bounds = (-100, 100)
# bestIndividual, bestFitness = mvo(bf.sphere, f1_bounds)
# print("Best individual:", bestIndividual)
# print("Best fitness:", bestFitness)
