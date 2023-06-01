import math
import random
import logging
import numpy as np
import pypsa
import GeneralFunctions as gf
from classBarra import Barra, Linha, No, Estrada
from typing import List
from random import randint
from copy import deepcopy


class Problema:
	def __init__(self, vet_barras: List[Barra], vet_linhas: List[Linha], vet_nos: List[No], vet_estradas: List[Estrada],
	             vet_posicaoNosBarras: List[int], mat_zonas: list, pot_demanda: float, tarifa_real_kwh: float,
	             multa_tensao: float, preco_cs: List[float], pot_cs: List[float], max_cs: int, alfa1: float, alfa2: float,
	             c2: float, pmin_cobertura: float, pmax: float, pmin: float, vmax: float, vmin: float):
		# Objetos
		self.Barras = vet_barras
		self.Linhas = vet_linhas
		self.NosTrafego = vet_nos
		self.Estradas = vet_estradas
		self.PosicaoNosBarras = vet_posicaoNosBarras
		self.Zonas = mat_zonas

		# Auxiliares
		self.FluxoNos = None
		self.calculaFluxoVeiculos()

		# Dados
		self.PotDemanda = pot_demanda
		self.PrecoEletroposto = preco_cs
		self.TarifaRealPorKWh = tarifa_real_kwh
		self.MultaDesvTensao = multa_tensao
		self.PotMaxBarras = pmax
		self.PotMinBarras = pmin
		self.V_nom = self.Barras[0].AbsV * 10 ** 3
		self.VdevInicial = 0
		self.PerdasInciais = 0
		self.VMax = vmax
		self.VMin = vmin
		self.PotEletroposto = pot_cs
		self.MaximoEletropostos = max_cs
		self.PMinCobertura = pmin_cobertura
		self.k1 = alfa1
		self.k2 = alfa2
		self.c2 = c2

	def calculaFluxoVeiculos(self):
		nBus = len(self.Barras)
		nRoads = len(self.Estradas)
		fluxoNos = np.zeros(nBus)

		for i in range(nRoads):
			i_index = self.PosicaoNosBarras[self.Estradas[i].i - 1] - 1
			j_index = self.PosicaoNosBarras[self.Estradas[i].j - 1] - 1
			# Add the road's flow to the nodes' flux
			fluxoNos[[i_index, j_index]] += self.Estradas[i].fluxo

		self.FluxoNos = fluxoNos

	def read_network_pf(self, network: pypsa.Network):
		nBus = len(self.Barras)

		# atualiza P e Q da barra do gerador
		self.Barras[0].P = network.buses_t.p['Bus 1'][0]
		self.Barras[0].Q = network.buses_t.q['Bus 1'][0]

		for i in range(0, nBus):
			self.Barras[i].V_pu = network.buses_t.v_mag_pu['Bus ' + str(i + 1)][0]
			self.Barras[i].AbsV = self.Barras[i].V_pu * self.V_nom
			self.Barras[i].Angulo = network.buses_t.v_ang['Bus ' + str(i + 1)][0]

	def calculaPerdasEDesvTensaoIniciais(self, network):
		df_powerflow = network.lines_t.p0.loc['now'] + network.lines_t.p1.loc['now']
		perdas = sum(abs(L) for L in df_powerflow) * 10 ** 3

		print(f'Perdas Inciais: {perdas:.2f} kW, era pra dar 203 kW')
		self.PerdasInciais = perdas  # em kW

		desvio = sum([1 - Vi for Vi in network.buses_t.v_mag_pu.loc['now']])*100 / (len(self.Barras) - 1)
		print(f'Desvio de V: {desvio:.2f}%, era pra dar 5.31%')

		soma_vdev = sum([(self.VMin - bus.V_pu)**2 * 10**6 for bus in self.Barras if bus.V_pu < self.VMin])
		self.VdevInicial = soma_vdev
		print('')

	def difAngular(self, line: Linha):
		return self.Barras[line.i - 1].Angulo - self.Barras[line.j - 1].Angulo


class Solucao:
	def __init__(self, dados_problema: Problema, dados_solucao: np.array):
		self.DadosProblema = dados_problema
		self.MatrizSolucao = dados_solucao  # Uma solução deve conter para cada eletroposto: Localização e Potência

		self.Barras = deepcopy(dados_problema.Barras)
		self.Linhas = deepcopy(dados_problema.Linhas)
		self.Perdas = [0]

		self.tornaFactivel()

		self.Network = None
		self.pypsa_update()
		# self.correnteRede()

	def pypsa_read(self):
		nBus = len(self.Barras)

		# atualiza P e Q da barra do gerador
		self.Barras[0].P = self.Network.buses_t.p['Bus 1'][0] * 10 ** 3  # Lê em MW, transforma em kW
		self.Barras[0].Q = self.Network.buses_t.q['Bus 1'][0] * 10 ** 3  # MW para kW

		for i in range(1, nBus):
			self.Barras[i].V_pu = self.Network.buses_t.v_mag_pu['Bus ' + str(i + 1)][0]
			self.Barras[i].AbsV = self.Barras[i].V_pu * self.DadosProblema.V_nom
			self.Barras[i].Angulo = self.Network.buses_t.v_ang['Bus ' + str(i + 1)][0]

		self.Perdas = self.Network.lines_t.p0.loc['now'] + self.Network.lines_t.p1.loc['now']

	def pypsa_update(self):
		self.Network = criaNetwork(self.MatrizSolucao, self.DadosProblema)
		self.Network.pf()
		self.pypsa_read()

	def obj_Perdas(self):
		# df_powerflow = self.Network.lines_t.p0.l   oc['now'] + self.Network.lines_t.p1.loc['now']
		perdas_tot = sum(abs(L) for L in self.Perdas) * 10 ** 3
		perdas_atual = perdas_tot - self.DadosProblema.PerdasInciais
		return perdas_atual * self.DadosProblema.TarifaRealPorKWh * 15 * 365    # 15 de 24 horas no dia, tomado como fator de utilização

	# soma_perdas = 0
	# for line in self.Linhas:
	# 	bus_i = self.Barras[line.i - 1]
	# 	bus_j = self.Barras[line.j - 1]
	# 	soma_perdas += line.G * (
	# 			bus_i.AbsV ** 2 + bus_j.AbsV ** 2 - 2 * bus_i.AbsV * bus_j.AbsV * math.cos(self.DadosProblema.difAngular(line)))
	# em W
	# self.perdas = soma_perdas * self.DadosProblema.TarifaRealPorKWh

	def obj_Vdev(self):
		soma_vdev = 0
		for bus in self.Barras:
			if bus.V_pu < self.DadosProblema.VMin:
				soma_vdev += (self.DadosProblema.VMin - bus.V_pu)**2 * self.DadosProblema.MultaDesvTensao

		return soma_vdev - self.DadosProblema.VdevInicial

	# # Versão Wang Adaptada
	# def obj_CoberturaDeTrafego(self):
	# 	cobertura = np.sum(
	# 		np.sum(self.MatrizSolucao * self.DadosProblema.PotEletroposto, axis=1) * self.DadosProblema.FluxoNos)
	# 	return cobertura

	# Wang Original
	def obj_CoberturaDeTrafego(self):
		# pMinCobertura = max(0.05, 1/len(self.Barras)) * self.DadosProblema.PotDemanda
		pMinCobertura = self.DadosProblema.PMinCobertura
		pot_barras = np.sum(self.MatrizSolucao * self.DadosProblema.PotEletroposto, axis=1)
		cobertura = np.sum(
				[fluxo for fluxo, pot_barra_i in zip(self.DadosProblema.FluxoNos, pot_barras) if pot_barra_i >= pMinCobertura])
		return cobertura

	def obj_CustoTotal(self):
		custo_eletr = np.sum(self.DadosProblema.PrecoEletroposto * np.sum(self.MatrizSolucao, axis=0))
		custo_instalacao = 500  # Custo Fixo
		custo_total = custo_eletr + custo_instalacao
		return custo_total

	@property
	def fitness(self):
		k1 = self.DadosProblema.k1
		k2 = self.DadosProblema.k2
		c2 = self.DadosProblema.c2
		fit = k1 * (self.obj_Perdas() + self.obj_Vdev() + self.obj_CustoTotal()) + k2 / (
					self.obj_CoberturaDeTrafego() + c2)
		return fit

	@property
	def fitnessA(self):
		fit = self.obj_Perdas() + self.obj_Vdev() + self.obj_CustoTotal()
		return fit

	@property
	def fitnessB(self):
		fit = self.obj_CoberturaDeTrafego()
		return fit

	@property
	def potTotalInstalada(self):
		return np.sum(self.DadosProblema.PotEletroposto * np.sum(self.MatrizSolucao, axis=0))

	def factivel(self):
		# Rest 1: Pot Instalada > Demanda
		if self.potTotalInstalada < self.DadosProblema.PotDemanda:
			self.fixDemanda()

		# # Rest 2: Potencia e Tensão nas barras
		# nBus = len(self.Barras)
		# for i in range(1, nBus):
		# 	pot_atual_pu = (self.Barras[i].P + self.MatrizSolucao[i]) / self.Barras[i].P
		# 	if (pot_atual_pu > self.DadosProblema.PotMaxBarras) or (pot_atual_pu < self.DadosProblema.PotMinBarras):
		# 		# print(f'A Potencia esta fora dos limites na barra {i}')
		# 		# self.fix2(i)
		# 		return False
		#
		# 	if (self.Barras[i].V_pu > self.DadosProblema.VMax) or (self.Barras[i].V_pu < self.DadosProblema.VMin):
		# 		print(f'A Tensao esta fora dos limites na barra {i}')
		# 		return False

		# Rest Corrente barras
		self.fixCorrenteMax()

		# Rest 3: Divisão em Zonas
		boolZonaComEletr = False
		for zone in self.DadosProblema.Zonas:
			for iBus in zone:
				if sum(self.MatrizSolucao[iBus - 1]) > 0:
					boolZonaComEletr = True
			if not boolZonaComEletr:
				# Possiblidade de fix, realizando um addEletroposto para uma barra dessa zona
				self.fixZonas()

		return True

	def tornaFactivel(self):
		# Rest Divisão em Zonas
		self.fixZonas()

		# Rest Pot Instalada > Demanda
		if self.potTotalInstalada < self.DadosProblema.PotDemanda:
			self.fixDemanda()

	#  # FUNÇÕES DE CORREÇÃO DE FALHAS DE RESTRIÇÃO
	def fixDemanda(self):
		while self.potTotalInstalada < self.DadosProblema.PotDemanda:
			iBus = randint(0, len(self.Barras) - 1)
			jEletr = randint(0, len(self.DadosProblema.PotEletroposto) - 1)
			if self.MatrizSolucao[iBus][jEletr] + 1 < self.DadosProblema.MaximoEletropostos:
				self.MatrizSolucao[iBus][jEletr] += 1

	def fixZonas(self):
		pmin = self.DadosProblema.PMinCobertura
		pot_barras = np.sum(self.MatrizSolucao * self.DadosProblema.PotEletroposto, axis=1)
		for zone in self.DadosProblema.Zonas:
			boolZonaComEletr = False
			for iBus in zone:
				if pot_barras[iBus-1] >= pmin:
					boolZonaComEletr = True
					break
			if not boolZonaComEletr:
				iBusInZone = random.choice(zone) - 1
				while pot_barras[iBusInZone] < pmin:
					jEletr = randint(0, len(self.DadosProblema.PotEletroposto) - 1)
					self.MatrizSolucao[iBusInZone][jEletr] += 1
					pot_barras[iBusInZone] += self.DadosProblema.PotEletroposto[jEletr]

	def fixCorrenteMax(self):
		Ilinhas = self.correnteRede()
		for j, iL in enumerate(Ilinhas):
			if iL > self.Linhas[j].Imax:
				print('Solução com corrente acima da máxima')

	def addEletroposto(self, iBus: int = -1, jEletr: int = -1):
		# Adiciona um eletroposto em uma barra aleatoria ou especificada
		if iBus == -1:
			iBus = randint(0, len(self.Barras) - 1)
		if jEletr == -1:
			jEletr = randint(0, len(self.DadosProblema.PotEletroposto) - 1)
		self.MatrizSolucao[iBus][jEletr] += 1

	def correnteRede(self):
		Ilines = []
		for linha in self.Linhas:
			# Calculate the current in the line
			line_name = "Line " + str(linha.i) + "-" + str(linha.j)
			sending_bus = self.Network.lines.loc[line_name, "bus0"]
			sending_voltage = self.Network.buses_t.v_mag_pu[sending_bus][0]
			line_power = self.Network.lines_t.p0[line_name][0]
			Ilines.append(line_power / sending_voltage)

		Ilines = np.array(Ilines) * 1e6 / self.DadosProblema.V_nom
		return Ilines


def criaNetwork(solucao: np.array, dados_problema: Problema):
	network = pypsa.Network()

	pot_barras = np.sum(solucao * dados_problema.PotEletroposto, axis=1)

	# Bus
	for barra in dados_problema.Barras:
		network.add("Bus", "Bus " + str(barra.i),
		            v_nom=dados_problema.V_nom/10**3,  # kV
		            control=barra.Tipo  # Slack, PV, PQ
		            )
		if barra.Tipo == 'SLACK':
			network.add("Generator", "Generator",
			            bus="Bus " + str(barra.i),
			            control=barra.Tipo
			            )
		else:
			network.add("Load", "Load " + str(barra.i),
			            bus="Bus " + str(barra.i),
			            p_set=(barra.P + pot_barras[barra.i-1]) / 10 ** 3,
			            q_set=barra.Q / 10 ** 3
			            )

	# Branch
	# Ainda falta passar pra p.u.
	for line in dados_problema.Linhas:
		network.add("Line", "Line " + str(line.i) + "-" + str(line.j),
		            bus0="Bus " + str(line.i),
		            bus1="Bus " + str(line.j),
		            r=line.R,
		            x=line.X
		            )
	return network
