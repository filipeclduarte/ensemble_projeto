import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils import series_to_supervised, gera_pool, pred_pool
from sklearn import preprocessing
from utils import NMSE
# usar o arg squared = True para calcular o RMSE
np.random.seed(123)
import pyswarms as ps
from skelm import ELMRegressor

# importando dados Santa Fe
df_sf = pd.read_csv('dados/df_santa_fe.csv')

# normalização gaussiana
scaler = preprocessing.StandardScaler().fit(df_sf['x'].values.reshape(-1,1))
df = scaler.transform(df_sf['x'].values.reshape(-1,1))

n_in = 10

pool_size = 100

treino = df[:1000]
teste = df[1000:]
X_teste_pred = df[(1000-n_in):].T # pegando as 10 últimas obs de treinamento até o final

# ### testando com 10 obs passadas
df_treino_sup = series_to_supervised(treino.reshape(-1,1), n_in=n_in)
X_treino = df_treino_sup.drop(columns='var1(t)').values
Y_treino = df_treino_sup['var1(t)'].values
Y_teste = np.copy(teste)
    
# # Treinar ELM
## Testando com 20 neurônios na camada escondida
# n_h = 20
# elm = ELMRegressor(n_neurons=n_h, ufunc='sigm')
# elm.fit(X_treino, Y_treino)
# Y_pred = np.empty(Y_teste.shape)
# Y_pred[0] = elm.predict(X_teste_pred[:,:n_in])
# for i in range(1,Y_teste.shape[0]):
#     X_teste_pred[:, n_in+i-1] = Y_pred[i-1]
#     Y_pred[i] = elm.predict(X_teste_pred[:, i:i+n_in])
    
# # ## RMSE de teste
# # retornar para a escala normal
# Y_pred_treino = elm.predict(X_treino)

# print('RMSE treino',mean_squared_error(Y_treino, Y_pred_treino, squared=True))
# print('NMSE teste', NMSE(Y_treino, Y_pred_treino))

# print('RMSE treino',mean_squared_error(Y_teste, Y_pred, squared=True))
# print('NMSE teste', NMSE(Y_teste, Y_pred))

Y_teste_desnorm = scaler.inverse_transform(Y_teste)
# Y_pred_teste_desnorm = scaler.inverse_transform(Y_pred)

# # # gráfico
# plt.plot(Y_teste_desnorm, label='Real output')
# plt.plot(Y_pred_teste_desnorm.mean(axis=1), label='Média')
# plt.plot(np.quantile(Y_pred_teste_desnorm, q=0.5, axis=1), label='Mediana')
# plt.legend()
# plt.grid()
# plt.show()

pool_size = 100
n_h_pool = [i for i in range(110, 210, 10)]
elm_pool = np.empty(10*len(n_h_pool), ELMRegressor)

for (i,n_h_i) in enumerate(n_h_pool):
    pool = np.array(gera_pool(pool_size, n_h_i, X_treino, Y_treino))
    pred_temp = [p.predict(X_treino) for p in pool]
    rmse_temp = [mean_squared_error(Y_treino, Y_pred_p, squared=True) for Y_pred_p in pred_temp]
    # best_p = np.argmin(rmse_temp)
    bests_p = np.argsort(rmse_temp)
    elm_pool[i*10:i*10+10] = pool[bests_p][:10]

# previsão do pool no treinamento
predictions_treino_pool = [p.predict(X_treino) for p in elm_pool]
# previsão do pool no teste
X_teste_pred_pool = np.copy(X_teste_pred)

predictions_teste_pool = pred_pool(elm_pool, n_in, Y_teste, X_teste_pred_pool)
predictions_treino_pool_desnorm = [scaler.inverse_transform(p) for p in predictions_treino_pool]
predictions_teste_pool_desnorm = [scaler.inverse_transform(p) for p in predictions_teste_pool]

# # # teste de plot de todas as previsões
# plt.plot(Y_teste_desnorm, label='Y_teste')
# [plt.plot(t) for t in predictions_teste_pool_desnorm[:5]]
# plt.legend()
# plt.show()


print('\nInicializar PSO')
print('--------------------')

Y_pred_teste_desnorm = np.array(predictions_teste_pool_desnorm).reshape(100, 10*len(n_h_pool)) # test_size, pool_size
# Y_pred_desnorm = np.array(predictions_treino_pool_desnorm).reshape(990, pool_size) # treinamento
Y_pred_treino = np.array(predictions_treino_pool).reshape(990, pool_size)

def weighted_average_ensemble(p):
    pnorm = p/p.sum()
    res = 1/Y_pred_treino.shape[0] * (pnorm * Y_pred_treino).sum(axis=1, keepdims=True)
	
    return res

def forward(pesos):
    Y_pred = weighted_average_ensemble(pesos)
    # loss = mean_squared_error(Y_treino, Y_pred, squared=True) 
    loss = NMSE(Y_treino, Y_pred)
    
    return loss

def f(x):
	"""
	Higher-level method to do the fitness in the whole swarm
	"""
	n_particles = x.shape[0]
	j = [forward(x[i]) for i in range(n_particles)]
	return np.array(j)

# inicializar swarm
options = {'c1': 1.49618, 'c2': 1.49618, 'w':0.7298}
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=10*len(n_h_pool), options=options)
# Perform optimization
cost, pos = optimizer.optimize(f, iters=100)

# aplicar pesos aos dados de teste
pesos_pso = np.array(pos)
# normalizar pesos
pesos_pso_norm = pesos_pso/pesos_pso.sum()

# calcular a previsão do Ensemble
Y_pred_teste = np.array(predictions_teste_pool).reshape(100, pool_size)
Y_pred_ensemble = 1/Y_pred_teste.shape[0] * (pesos_pso_norm * Y_pred_teste).sum(axis=1, keepdims=True)

print('Y_pred_ensemble:', Y_pred_ensemble[:5])
print('Y_teste: ', Y_teste[:5])

# calcular o erro
loss = mean_squared_error(Y_teste, Y_pred_ensemble, squared=True) # rmse
nmse_loss = NMSE(Y_teste, Y_pred_ensemble)

print('Ensemble')
print('RMSE: ', loss)
print('NMSE: ', nmse_loss)

print('-------')
print('Média')
print('RMSE: ', mean_squared_error(Y_teste, Y_pred_teste.mean(axis=1), squared=True))
print('NMSE: ', NMSE(Y_teste, Y_pred_teste.mean(axis=1)))

print('------ ')
print('Mediana')
print('RMSE: ', mean_squared_error(Y_teste, np.quantile(Y_pred_teste, 0.5, axis=1), squared=True))
print('NMSE: ', NMSE(Y_teste, np.quantile(Y_pred_teste,0.5, axis=1)))


# gráfico
plt.plot(Y_teste_desnorm, label='Real output')
plt.plot(scaler.inverse_transform(Y_pred_ensemble), label='Ensemble output')
plt.plot(Y_pred_teste_desnorm.mean(axis=1), label='Média')
plt.plot(np.quantile(Y_pred_teste_desnorm, q=0.5, axis=1), label='Mediana')
plt.legend()
plt.grid()
plt.show()

#TODO: criar 10 ELM para para cada configuração de neurônios na camada escondida de 10 até 100 em 10 e 10
#TODO: criar modelos de 10 até 200 na camada escondida
#TODO: criar diversos modelos variando a quantidade de neurôinos na camada escondida
#TODO: criar o gráfico mostrando o erro em relação à quantidade de ELM do ensemble
#TODO: testando PSO para cada ensemble aumentando um ELM por vez. 
#TODO: salvar tudo e tentar reproduzir os resultados
