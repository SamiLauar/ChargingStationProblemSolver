from classBarra import Barra, Linha, No, Estrada
from typing import List
import pandas as pd
import numpy as np
import csv
import pypsa


def compute_bus_data(filename: str, network: pypsa.Network):
	"""

    :param network: PYPSA
    :type filename: object
    """
	bus_list = list()
	with open('data/' + filename, 'r', newline='') as file:
		reader = csv.reader(file)
		for row in reader:
			line = str(row[0]).split(sep=';')
			# print(line)
			if line[0].isnumeric():
				# Ainda falta passar pra p.u.
				bus_list.append(Barra(tipo=line[1], posicao=int(line[0]), abs_v=float(line[4]), angulo_v=0, p=float(line[2]),
				                      q=float(line[3])))

				network.add("Bus", "Bus " + line[0],
				            v_nom=float(line[4]),   # kV
				            control=line[1]         # Slack, PV, PQ
				            )

				if line[1] == "SLACK":
					network.add("Generator", "Generator",
					            bus="Bus " + line[0],
					            control=line[1]
					            )
				else:
					network.add("Load", "Load " + line[0],
					            bus="Bus " + line[0],
					            p_set=float(line[2]) / 10**3,
					            q_set=float(line[3]) / 10**3)
	return bus_list, network


def compute_branch_data(filename: str, bus_list: list, network: pypsa.Network):
	"""

    :param network: PYPSA
    :param filename:
    :type bus_list: object
    """
	branch_list = list()
	with open('data/' + filename, 'r', newline='') as file:
		reader = csv.reader(file)
		for row in reader:
			line = str(row[0]).split(sep=';')
			# print(line)
			if line[0].isnumeric():
				# Ainda falta passar pra p.u.
				branch_list.append(Linha(barra_i=bus_list[int(line[0]) - 1], barra_j=bus_list[int(line[1]) - 1],
				                         r_ij=float(line[2]), x_ij=float(line[3]), imax=float(line[4])))
				network.add("Line", "Line " + line[0] + "-" + line[1],
				            bus0="Bus " + line[0],
				            bus1="Bus " + line[1],
				            r=float(line[2]),
				            x=float(line[3])
				            )
	return branch_list, network


def compute_traffic_data(filename: str):
	"""

    :param filename:
    """
	traffic_node_list = list()
	road_list = list()
	aux = 0
	with open('data/' + filename, 'r', newline='') as file:
		reader = csv.reader(file)
		for row in reader:
			line = str(row[0]).split(sep=';')
			if int(line[0]) == aux + 1:
				traffic_node_list.append(No(posicao=int(line[0]), peso=float(line[3])))
				aux = aux + 1

		file.seek(0)  # move cursor back to beginning of file
		reader = csv.reader(file)

		for row in reader:
			line = str(row[0]).split(sep=';')
			# Ainda falta passar pra p.u.
			road_list.append(Estrada(no_i=traffic_node_list[int(line[0]) - 1], no_j=traffic_node_list[int(line[1]) - 1],
			                         comprimento=float(line[2])))
	return [traffic_node_list, road_list]


def compute_traffic_node_data(filename: str):
	"""

    :param filename:
    """
	traffic_node_list = list()
	with open('data/' + filename, 'r', newline='') as file:
		reader = csv.reader(file)
		for row in reader:
			line = str(row[0]).split(sep=';')
			traffic_node_list.append(No(posicao=int(line[0]), peso=float(line[1])))
	return traffic_node_list


def compute_traffic_road_data(filename: str, traffic_node_list: List[No]):
	"""

    :param traffic_node_list:
    :param filename:
    """
	road_list = list()
	with open('data/' + filename, 'r', newline='') as file:
		reader = csv.reader(file)
		for row in reader:
			line = str(row[0]).split(sep=';')
			# Ainda falta passar pra p.u.
			road_list.append(Estrada(no_i=traffic_node_list[int(line[0]) - 1], no_j=traffic_node_list[int(line[1]) - 1],
			                         comprimento=float(line[2])))
	return road_list
