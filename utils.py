import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from ELM import *

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#TODO: implementar função que gera o pool de ELM

# def gera_pool(N, n_h, X, Y):
# 	pool = []
# 	for _ in range(N):
# 		mod = ELMRegressor(n_h)
# 		mod.fit(X,Y)
# 		pool.append(mod)
# 	return pool

def gera_pool(N, n_h, X, Y):
	pool = []
	step = int(Y.shape[0] / N)
	for i in range(N):
		mod = ELMRegressor(n_h)
		mod.fit(X[i:i+step,:], Y[i:i+step])
		pool.append(mod)
	return pool

def pred_pool(pool, n_in, Y_teste, X_teste):
	predictions = []
	for p in pool:
		X_teste_pred = np.copy(X_teste)
		Y_pred = np.empty(Y_teste.shape)
		Y_pred[0] = p.predict(X_teste_pred[:,:n_in])

		for i in range(1,Y_teste.shape[0]):
			X_teste_pred[:, n_in+i-1] = Y_pred[i-1]
			Y_pred[i] = p.predict(X_teste_pred[:, i:i+n_in])
			# X_teste_temp = X_teste_pred[:,i:i+n_in]
			# Y_pred[i] = p.predict(X_teste_temp)
		predictions.append(Y_pred)

	return predictions
	

#TODO: Implementar função que será otimizada pelo algoritmo PSO
#### Essa função vai pegar os pesos, normalizá-los (wi_norm = wi/W), calcular a média ponderada e o erro RMSE


#TODO: função  normalized mean square error (NMSE)
def NMSE(Y, Y_pred):
	# return (1/Y.var()) * ((Y - Y_pred)**2).mean()
	return mean_squared_error(Y, Y_pred)/Y.var()

