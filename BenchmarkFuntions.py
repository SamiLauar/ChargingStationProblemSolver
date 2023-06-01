import random
import numpy as np


# Define the benchmark functions
def sphere(x):
	return sum(xi ** 2 for xi in x)


def ackley(x):
	d = len(x)
	term1 = -20 * np.exp(-0.2 * np.sqrt(sum(xi ** 2 for xi in x) / d))
	term2 = -np.exp(sum(np.cos(2 * np.pi * xi) for xi in x) / d)
	return term1 + term2 + 20 + np.exp(1)


def rosenbrock(x):
	d = len(x)
	return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(d - 1))


def rastrigin(x):
	d = len(x)
	return 10 * d + sum(xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x)


def griewank(x):
	d = len(x)
	sum_term = sum(xi ** 2 / 4000 for xi in x)
	prod_term = 1
	for i in range(d):
		prod_term *= np.cos(x[i] / np.sqrt(i+1))
	return sum_term - prod_term + 1


def schwefel(x):
	d = len(x)
	return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


