from __future__ import division
from pylab import *
import sys
import math
import numpy.random as rd
import numpy.linalg as la
import numpy as np
import scipy as sp
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import norm
import statsmodels.api as sm




#indicator function for the interval [a,b]
def I(x, a, b):
	if a<= x < b:
		return 1
	else:
		return 0

def iv_2sls(y, x, z):
	y = np.matrix(y).T
	x = np.matrix(x).T
	z = np.matrix(z).T

	p_z = z*la.inv(z.T*z)*z.T
	x_z = p_z*x

	beta = la.inv(x_z.T*x_z)*x_z.T*y
	var = la.inv(x_z.T*x_z)

	return {'beta': beta.item(0), 'var': var.item(0)}


def confidence_interval(this_beta, var, alpha, beta_real = .15):
	bound = norm.ppf(1-alpha/2)
	c_low = this_beta - var**(1/2)*bound
	c_high = this_beta + var**(1/2)*bound
	contain_0 = (0>= c_low)*(0 <= c_high)
	contain_beta = (beta_real >= c_low)*(beta_real <= c_high)
	length = c_high - c_low

	result = np.array([contain_0,contain_beta,length])
	return result

def stats(alpha,n,rho =.9, beta = .15):

	w = rd.random(n)
	cov_er = [[1,rho],[rho,1]]

	z = -.5*1*(w<.2)-.1*1*(.2 <= w)*(w <.4)+.1*1*(4 <= w)*(w <6)+1*(.6 < w)


	ers = rd.multivariate_normal([0,0],cov_er,n)
	epsilon = ers[:,0]

	v = ers[:,0]
	u = (1+z)*epsilon

	x = 4*z**2 + v
	y = beta*x+u
	#----Inefficient 2sls
	model_2sls = iv_2sls(y,x,z)
	beta_2sls = model_2sls['beta']
	var_2sls = model_2sls['var']

	#----- Using efficient (infeasible) weight matrix -----
	gn = 4*z**2/(cov_er[0][0]*(1+z)**2)

	model_efficient = iv_2sls(y,x,gn)
	beta_efficient = model_efficient['beta']
	var_efficient = model_efficient['var']


	#------Estimating efficient instruments -----------
	u_hat  = y - beta_2sls*x

	z_J = np.unique(z)
	z_counts = []
	E_z_x = []
	E_z_u2 = []
	for i in range(len(z_J)):
		sum_i = np.sum(z==z_J[i])
		z_counts.append(sum_i)
		x_sum = np.sum(x[z == z_J[i]])
		u_sum = np.sum(u_hat[z == z_J[i]]**2)
		E_z_x.append(x_sum/sum_i)
		E_z_u2.append(u_sum/sum_i)

	E_z_u2 = np.array(E_z_u2)
	E_z_x = np.array(E_z_x)
	weights = E_z_x/E_z_u2

	g_hat = np.zeros(n)
	for i in range(len(z_J)):
		g_hat += weights[i]*(z == z_J[i])

	model_eff_hat = iv_2sls(y,x,g_hat)
	beta_eff_hat = model_eff_hat['beta']
	var_eff_hat = model_eff_hat['var']

	betas = np.array([beta_2sls,beta_efficient, beta_eff_hat])
	var = np.array([var_2sls, var_efficient, var_eff_hat])

	return confidence_interval(betas, var, alpha)


def main():

	n = 100
	rho = .9
	beta = .15
	reps = 10000
	

	rep_stats_10 = stats(.1,n)
	for i in range(reps -1):
	 	rep_stats_10+= stats(.1,n)

	rep_stats_5 = stats(.05,n)
	for i in range(reps -1):
	 	rep_stats_5+= stats(.05,n)

	rep_stats_1 = stats(.01,n)
	for i in range(reps -1):
	 	rep_stats_1+= stats(.01,n)


	stats_10 = rep_stats_10/reps
	stats_5 = rep_stats_5/reps
	stats_1 = rep_stats_1/reps

	alphas = np.array([[.1],[.05],[.01]])
	tables = []
	for i in range(3):
		row_1 = np.concatenate((alphas[0], stats_10[i]), axis = 1)
		row_2 = np.concatenate((alphas[1], stats_5[i]), axis = 1)
		row_3 = np.concatenate((alphas[2], stats_1[i]), axis = 1)
		table = PrettyTable(['alpha', '2sls', 'Exact GMM', 'Estimated GMM'])
		table.padding_width = 1
		table.add_row(row_1)
		table.add_row(row_2)
		table.add_row(row_3)
		tables.append(table)

	print tables[0]
	print tables[1]
	print tables[2]


if __name__ == "__main__":
    main()



