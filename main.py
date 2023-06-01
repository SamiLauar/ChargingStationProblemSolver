"""
Name: MVO Optimizer
Copyright: Sami Nasser Lauar
Author: Sami Nasser Lauar
Start Date: 25/03/2023 14:00
End Date:
Description: Solving Charging Station Placement Problem
"""

# Resolvendo o Problema de Alocação e Dimensionamento de Eletropostos
from IPython.display import display
from classBarra import Barra, Linha, No, Estrada
from ProblemClass import Problema, Solucao
import GeneralFunctions as gfunc
import SolverFunctions as solver
import MVO
import DE
import CSA
from random import random
import pandas as pd
import numpy as np
import csv
import pypsa
import logging
import time
import datetime

if __name__ == '__main__':

	# PyPsa Silence e print config 2 casas decimais
	logging.getLogger("pypsa").setLevel(logging.WARNING)
	np.set_printoptions(precision=3)

	# # Variáveis
	max_cs = 4  # num max de eletropostos em uma barra, Ou, 15% dos eletropostos numa só barra, por exemplo
	vet_demanda = [500, 650, 800, 1000]  # em kW
	tarifa_kWh = 0.2        # Preço da energia ($/kWh)
	penalidade_tensao = 2e6  # Penalidade aplicada por Barra com Tensao abaixo do limite
	pot_cs = (20, 50)       # kW: potencia unitaria dos eletropostos
	preco_cs = (2400, 5000)  # Obtido a partir de um preço aprox de um eletroposto de 20 kW ($/kW de eletroposto)
	pmin_cobertura = 50

	p_min, p_max = 0.9, 1.2  # em p.u, das linhas?
	v_min, v_max = 0.93, 1.05  # em p.u, 0.93 para o cenario brasileiro (PRODIST)

	# Parâmetros das Meta Heuristicas
	max_iter = 120
	max_tempo = 1000
	pop_de = 25
	pop_mvo = 20
	n_testes = 4
	cr = 0.7
	sf = 0.7

	# Variveis da Função MultiObjetivo de Fitness
	alfa1 = 0.00001
	alfa2 = 2
	c2 = 0.01

	# # Leitura dos dados e criação das variaveis
	network_base = pypsa.Network()
	[lista_barras, network_base] = gfunc.compute_bus_data('ieee33_bus_data.csv', network_base)
	[lista_linhas, network_base] = gfunc.compute_branch_data('ieee33_branch_data.csv', lista_barras, network_base)
	lista_nos = gfunc.compute_traffic_node_data('25node_data.csv')
	lista_estradas = gfunc.compute_traffic_road_data('roads_data.csv', lista_nos)

	df_posicao = pd.read_csv('data/' + 'ieee33_and_traffic25node.csv', sep='\t', header=0)
	df_zonas = pd.read_csv('data/' + 'ieee33_zones_division.csv', sep=';', header=0)
	vet_posicao_nos_barras = df_posicao['ieee33busRDS'].tolist()

	mat_zonas = [[] for _ in range(df_zonas.max()['ZoneNum'] + 1)]
	for index, row in df_zonas.iterrows():
		mat_zonas[row['ZoneNum']].append(row['Bus'])

	# Criacao do Objeto para o Problema
	problema_teste = Problema(lista_barras, lista_linhas, lista_nos, lista_estradas, vet_posicao_nos_barras, mat_zonas,
	                          pot_demanda=0, tarifa_real_kwh=tarifa_kWh, multa_tensao=penalidade_tensao,
	                          preco_cs=preco_cs, pot_cs=pot_cs, max_cs=max_cs, alfa1=alfa1, alfa2=alfa2, c2=c2,
	                          pmin_cobertura=pmin_cobertura, pmax=p_max, pmin=p_min, vmax=v_max, vmin=v_min)

	# Calculos do PyPSA PowerFlow, PerdasInicias e DesvioDeTensaoInicial
	network_base.pf()
	problema_teste.read_network_pf(network_base)  # Atualiza P e Q da barra Slack e Vmag e Vang das barras PQ
	problema_teste.calculaPerdasEDesvTensaoIniciais(
			network_base)  # Calcula as perdas antes de acrescentar os eletropostos

	# tensoes = [[bus.i, bus.V_pu] for i, bus in enumerate(problema_teste.Barras)]
	# solver.excelGraphPlot(tensoes, 'V_Inicial', 'pot-0')

	for demanda in vet_demanda:

		for test in range(n_testes):
			print('\nInício  ------------------------------- \n')
			print(f'Demanda:  \033[33m{demanda}\033[m /  Teste {test + 1}')
			start_time = time.time()

			problema_teste.PotDemanda = demanda

			# # Solução

			# MVO SOLVE <-----------------------------------------------------------------------------------------------
			str_algorithm = 'MVO'
			best_solution, population, resultados = MVO.solve(problema_teste, pop_mvo, bounds=(0, max_cs),
			                                                  max_iter=max_iter, max_tempo=max_tempo)
			str_parametros = f''
			solver.resultsAnalysis(best_solution, population, resultados, problema_teste, start_time, str_algorithm,
			                       str_parametros, max_iter, max_tempo)

			# DE SOLVE <------------------------------------------------------------------------------------------------
			str_algorithm = 'DE'
			best_solution, population, resultados = DE.solve(problema_teste, pop_size=pop_de, bounds=(0, max_cs),
			                                                 max_generations=max_iter, max_tempo=max_tempo)
			str_parametros = f'crossover_prob={cr} / scale_factor={sf}'
			solver.resultsAnalysis(best_solution, population, resultados, problema_teste, start_time, str_algorithm,
			                       str_parametros, max_iter, max_tempo)

			# Hybrid DE-MVO SOLVE <--------------------------------------------------------------------------------------
			str_algorithm = 'DE-MVO'
			best_solution, population, resultados = DE.hybridMVO(problema_teste, pop_size=pop_mvo, bounds=(0, max_cs),
			                                                     max_generations=max_iter, max_tempo=max_tempo,
			                                                     crossover_prob=cr, scale_factor=sf)
			str_parametros = f'crossover_prob={cr} / scale_factor={sf}'
			solver.resultsAnalysis(best_solution, population, resultados, problema_teste, start_time, str_algorithm,
			                       str_parametros, max_iter, max_tempo)

		# CSA SOLVE <-----------------------------------------------------------------------------------------------
		# str_algorithm += 'CSA'
		# best_solution, population, resultados = CSA.solve(population=population, bounds=(0, max_cs), max_iter=max_iter)

		# # Analise e Print dos Resultados

		# print('\nFim  ------------------------------- \n')
		# tempo = time.time() - start_time
		# print(f'Time Elapsed: \033[33m{tempo:.1f}\033[ms')
		#
		# # Print no Python Console
		# solver.SolucaoPrint(best_solution)
		# solver.FitnessPrint(population)
		# population_fitness = np.array([sol.fitness for sol in population])
		# solver.ObjMediaPrint(population, np.argmin(population_fitness))
		# best_fitness = best_solution.fitness
		# melhora = (resultados[0][1] - best_fitness) / resultados[0][1] * 100
		# print(
		# 		f'Resultado: {resultados[0][1]:.3f} para \033[31m{best_fitness:.3f}\033[m \n   Melhora de \033[34m{melhora:.1f}\033[m%')
		#
		# tensoes = [[bus.i, bus.V_pu] for i, bus in enumerate(best_solution.Barras)]
		# solver.excelGraphPlot(tensoes, str_algorithm + '_V_lastRunResults',
		#                       str(round(best_fitness, 3)) + '_pot-' + str(demanda))
		# desvio = [1 - vpu[1] for vpu in tensoes]
		#
		# # vet das Colunas para Excel
		# vetResultados = [
		# 		str_algorithm,
		# 		problema_teste.PotDemanda,
		# 		best_solution.potTotalInstalada,
		# 		str(round(best_fitness, 3)).replace(".", ","),
		# 		round(resultados[0][1], 3),
		# 		round(melhora, 1),
		# 		round(tempo, 1),
		# 		round(best_solution.obj_Perdas(), 1),
		# 		round(best_solution.obj_Vdev(), 1),
		# 		round(best_solution.obj_CustoTotal(), 1),
		# 		str(round(best_solution.obj_CoberturaDeTrafego(), 3)).replace(".", ","),
		# 		best_solution.obj_Perdas() + best_solution.obj_Vdev() + best_solution.obj_CustoTotal(),
		# 		sum(desvio) / (len(desvio) - 1),
		# 		f'alfa1={alfa1} / alfa2={alfa2} / c2={c2}',
		# 		f'tarifa={tarifa_kWh} / penalidade={penalidade_tensao} / max_cs={max_cs}',
		# 		f'PopSize={size_population}, MaxIter={max_iter}, MaxTempo={max_tempo}',
		# 		str_parametros,
		# 		solver.SolucaoString(best_solution)]
		#
		# solver.excelAddRow(vetResultados, str_algorithm + '_Results',
		#                    'Results')  # 'pot-' + str(problema_teste.PotDemanda))
		#
		# solver.excelGraphPlot(resultados, str_algorithm + '_lastRunResults',
		#                       'pot-' + str(problema_teste.PotDemanda))
		#
		# for ind in population:
		# 	vet_FObj = [
		# 			ind.potTotalInstalada,
		# 			round(ind.fitness, 3),
		# 			round(ind.fitnessA, 1),
		# 			round(ind.fitnessB, 3),
		# 			round(ind.obj_Perdas(), 1),
		# 			round(ind.obj_Vdev(), 1),
		# 			round(ind.obj_CustoTotal(), 1),
		# 			solver.SolucaoString(ind)]
		# 	solver.excelAddRow(vet_FObj, str_algorithm + '_Pareto_Results', 'Results')
