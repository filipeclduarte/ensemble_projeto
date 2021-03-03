import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils import series_to_supervised, gera_pool, pred_pool
from sklearn import preprocessing
from utils import NMSE
# usar o arg squared = True para calcular o RMSE
np.random.seed(123)

# importar ELM
from ELM import * # importando sigmoid e a classe ELMRresgressor
# PSO 
# Import PySwarms
import pyswarms as ps

# importando dados Santa Fe
df_sf = pd.read_csv('dados/df_santa_fe.csv')

# normalização gaussiana
scaler = preprocessing.StandardScaler().fit(df_sf['x'].values.reshape(-1,1))
df = scaler.transform(df_sf['x'].values.reshape(-1,1))

#TODO: fazer n_in = 10, 20, ..., 90, 100
# n_ins = [i for i in range(10, 110, 10)]

# for n_in in n_ins:
#     print('n_in: ', n_in)
#     treino = df[:1000]
#     teste = df[1000:]
#     X_teste_pred = df[(1000-n_in):].T # pegando as 10 últimas obs de treinamento até o final

#     # Estruturar os dados
#     ## transformar o problema de série em supervised learning
#     ### testando com n_in obs passadas
#     df_treino_sup = series_to_supervised(treino.reshape(-1,1), n_in=n_in)
#     X_treino = df_treino_sup.drop(columns='var1(t)').values
#     Y_treino = df_treino_sup['var1(t)'].values

#     df_teste_sup = series_to_supervised(teste.reshape(-1,1), n_in=n_in)
#     X_teste = df_teste_sup.drop(columns='var1(t)').values
#     Y_teste = df_teste_sup['var1(t)'].values

#     # Treinar ELM
#     ## Testando com 20 neurônios na camada escondida
#     n_h = 20
#     elm = ELMRegressor(n_h)
#     elm.fit(X_treino, Y_treino)

#     # Previsões
#     # primeira pred
#     Y_pred = np.empty(Y_teste.shape)
#     Y_pred[0] = elm.predict(X_teste_pred[:,:n_in])
#     X_teste_pred_pool = np.copy(X_teste_pred)
#     for i in range(1,Y_teste.shape[0]):
#         X_teste_pred[:, n_in+i-1] = Y_pred[i-1] 
#         X_teste_temp = X_teste_pred[:,i:i+n_in]
#         Y_pred[i] = elm.predict(X_teste_temp)

#     ## RMSE de teste
#     # retornar para a escala normal
#     Y_teste_desnorm = scaler.inverse_transform(Y_teste)
#     Y_pred_desnorm = scaler.inverse_transform(Y_pred)

#     print(Y_teste_desnorm)
#     print(Y_pred_desnorm)

#     RMSE = mean_squared_error(Y_teste_desnorm.reshape(-1,1), Y_pred_desnorm.reshape(-1,1), squared = True)
#     print(f'RMSE = {RMSE}')

#     ## Plotar 
#     plt.plot(Y_teste_desnorm.reshape(-1,1), label='Real')
#     plt.plot(Y_pred_desnorm.reshape(-1,1), label='Prediction')
#     plt.legend()
#     plt.title(f'Teste {n_in}')
#     plt.show()

#     # Utilizar a função gera_pool
#     pool_size = 100
#     elm_pool = gera_pool(pool_size, n_h, X_treino, Y_treino)
#     predictions_pool = pred_pool(elm_pool, n_in, Y_teste, X_teste_pred_pool)

#     # scaler inverse
#     predictions_pool_desnorm = [scaler.inverse_transform(p) for p in predictions_pool]

#     # calcular RMSE
#     RMSE_pool = np.array([mean_squared_error(Y_teste_desnorm.reshape(-1,1), p.reshape(-1,1), squared = True) for p in predictions_pool_desnorm])
#     # print('RMSE_pool\n', RMSE_pool)
#     print('Argmin: ', RMSE_pool.argmin())
#     print('RMSE min: ', RMSE_pool.min())
#     print(f'{n_in} RMSE médio: ', RMSE_pool.mean())
#     print('RMSE std: ', RMSE_pool.std(ddof=1))
n_in = 10

## treinamento as 1000 primeiras obs
treino = df[:1000]
teste = df[1000:]
X_teste_pred = df[(1000-n_in):].T # pegando as 10 últimas obs de treinamento até o final

# Estruturar os dados
## transformar o problema de série em supervised learning
### testando com 10 obs passadas
df_treino_sup = series_to_supervised(treino.reshape(-1,1), n_in=n_in)
X_treino = df_treino_sup.drop(columns='var1(t)').values
Y_treino = df_treino_sup['var1(t)'].values

df_teste_sup = series_to_supervised(teste.reshape(-1,1), n_in=n_in)
X_teste = df_teste_sup.drop(columns='var1(t)').values
# Y_teste = df_teste_sup['var1(t)'].values
Y_teste = np.copy(teste)
    
# Treinar ELM
## Testando com 20 neurônios na camada escondida
n_h = 20
# elm = ELMRegressor(n_h)
# elm.fit(X_treino, Y_treino)

# Previsões
# primeira pred
# Y_pred = np.empty(Y_teste.shape)
# Y_pred[0] = elm.predict(X_teste_pred[:,:n_in])
X_teste_pred_pool = np.copy(X_teste_pred)
# for i in range(1,Y_teste.shape[0]):
#     X_teste_pred[:, n_in+i-1] = Y_pred[i-1] 
#     X_teste_temp = X_teste_pred[:,i:i+n_in]
#     Y_pred[i] = elm.predict(X_teste_temp)

## RMSE de teste
# retornar para a escala normal
Y_treino_desnorm = scaler.inverse_transform(Y_treino)
Y_teste_desnorm = scaler.inverse_transform(Y_teste)
# Y_pred_desnorm = scaler.inverse_transform(Y_pred)

# print(Y_teste_desnorm)
# print(Y_pred_desnorm)

# RMSE = mean_squared_error(Y_teste_desnorm.reshape(-1,1), Y_pred_desnorm.reshape(-1,1), squared = True)
# nmse = NMSE(Y_teste_desnorm.reshape(-1,1), Y_pred_desnorm.reshape(-1,1))
# print(f'RMSE = {RMSE}')
# print(f'NMSE = {nmse}')

## Plotar 
# plt.plot(Y_teste_desnorm.reshape(-1,1), label='Real')
# plt.plot(Y_pred_desnorm.reshape(-1,1), label='Prediction')
# plt.legend()
# plt.title('Teste')
# plt.show()

# Utilizar a função gera_pool
pool_size = 100
elm_pool = gera_pool(pool_size, n_h, X_treino, Y_treino)
# previsão do pool no treinamento
predictions_treino_pool = [p.predict(X_treino) for p in elm_pool]
# previsão do pool no teste
predictions_teste_pool = pred_pool(elm_pool, n_in, Y_teste, X_teste_pred_pool)

# scaler inverse da previsao
predictions_treino_pool_desnorm = [scaler.inverse_transform(p) for p in predictions_treino_pool]
predictions_teste_pool_desnorm = [scaler.inverse_transform(p) for p in predictions_teste_pool]

# # teste de plot de todas as previsões
# plt.plot(Y_teste_desnorm, label='Y_teste')
# [plt.plot(t) for t in predictions_teste_pool_desnorm[:5]]
# plt.legend()
# plt.show()

# predictions_pool_mean = np.mean(predictions_pool_desnorm, axis=0)
# predictions_pool_median = np.quantile(predictions_pool_desnorm, 0.5)

# calcular RMSE
# RMSE_pool = np.array([mean_squared_error(Y_teste_desnorm.reshape(-1,1), p.reshape(-1,1), squared = True) for p in predictions_pool_desnorm])
# NMSE_pool = np.array([NMSE(Y_teste_desnorm.reshape(-1,1), p.reshape(-1,1)) for p in predictions_pool_desnorm])
# # print('RMSE_pool\n', RMSE_pool)
# print('RMSE Argmin: ', RMSE_pool.argmin())
# print('RMSE min: ', RMSE_pool.min())
# print('RMSE médio: ', RMSE_pool.mean())
# print('RMSE mediano: ', np.quantile(RMSE_pool, 0.5))
# print('RMSE std: ', RMSE_pool.std(ddof=1))
# print('-----------------------------------')
# print('NMSE Argmin: ', NMSE_pool.argmin())
# print('NMSE min: ', NMSE_pool.min())
# print('NMSE médio: ', NMSE_pool.mean())
# print('NMSE mediano: ', np.quantile(NMSE_pool,0.5))
# print('NMSE std: ', NMSE_pool.std(ddof=1))

print('\nInicializar PSO')
print('--------------------')

Y_pred_teste_desnorm = np.array(predictions_teste_pool_desnorm).reshape(100, pool_size) # test_size, pool_size
# Y_pred_desnorm = np.array(predictions_treino_pool_desnorm).reshape(990, pool_size) # treinamento
Y_pred_treino = np.array(predictions_treino_pool).reshape(990, pool_size)

def weighted_average_ensemble(p):
    pnorm = p/p.sum()
    res = 1/Y_pred_treino.shape[0] * (pnorm * Y_pred_treino).sum(axis=1, keepdims=True)
	
    return res

def forward(pesos):
    Y_pred = weighted_average_ensemble(pesos)
    loss = mean_squared_error(Y_treino, Y_pred, squared=True)  
    
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
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=pool_size, options=options)
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

 # test_size, pool_size
#TODO: Ordenar ELM pelo erro

#TODO: Adicionar um modelo por vez ao ensemble. a cada entrada de modelo, otimizar com PSO


# iniciar um laço de repetição para testar o enesemble para 1, 2, ..., N modelos
## Utilizar o PSO para otimizar os pesos dos modelos ELM na combinação (média ponderada)

# Realizar previsão h+1, incorpora na série e prevê h+2, ..., até o último ponto da série