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

n_in = 10
n_inputs = [i for i in range(10, 110, 10)]
n_hidden = [i for i in range(10, 200, 10)]
pool_size = 100

## treinamento as 1000 primeiras obs
treino = df[:1000]
teste = df[1000:]
X_teste_pred = df[(1000-n_in):].T # pegando as 10 últimas obs de treinamento até o final

def treinamento(serie_treino, n_inputs, n_hidden, pool_size):

    pool = np.empty(shape=(len(n_inputs),pool_size), dtype=ELMRegressor)
    pesos_pso = []
    cost = []
    for (_, n_in) in enumerate(n_inputs):
        print('n_in: ', n_in)
        df_treino_sup = series_to_supervised(serie_treino.reshape(-1,1), n_in=n_in)
        X_treino = df_treino_sup.drop(columns='var1(t)').values
        Y_treino = df_treino_sup['var1(t)'].values
        
        
        for (j, n_h) in enumerate(n_hidden):
            print('n_h: ', n_h)
            pool_temp = np.array(gera_pool(pool_size, n_h, X_treino, Y_treino))
            
            predictions_treino_pool = [p.predict(X_treino) for p in pool_temp]
            # predictions_treino_pool_desnorm = [scaler.inverse_transform(p) for p in predictions_treino_pool]
            Y_pred_treino = np.array(predictions_treino_pool).reshape(-1, pool_size)
            rmse_temp = [mean_squared_error(Y_treino, Y_pred_p, squared=True) for Y_pred_p in predictions_treino_pool]
            bests_p = np.argsort(rmse_temp)
            # elm_pool[n_h_i - 10:n_h_i] = pool[bests_p][:10]
            pool[j, j:j+10] = pool_temp[bests_p][:10]

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

            options = {'c1': 1.49618, 'c2': 1.49618, 'w':0.7298}
            optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=pool_size, options=options)
            cost_temp, pos = optimizer.optimize(f, iters=100)
            cost.append(cost_temp)
            pesos_pso_temp = np.array(pos)
            pesos_pso_norm = pesos_pso_temp/pesos_pso_temp.sum()
            pesos_pso.append(pesos_pso_norm)

    return pool, np.array(pesos_pso), np.array(cost)

elm_pool, pesos_pso, loss = treinamento(treino, n_inputs, n_hidden, pool_size)

print('elm_pool')
print(elm_pool)
print('loss')
print(loss)
np.savetxt('resultados_loss.txt', loss)
# # Estruturar os dados
# ## transformar o problema de série em supervised learning
# ### testando com 10 obs passadas
# df_treino_sup = series_to_supervised(treino.reshape(-1,1), n_in=n_in)
# X_treino = df_treino_sup.drop(columns='var1(t)').values
# Y_treino = df_treino_sup['var1(t)'].values

# df_teste_sup = series_to_supervised(teste.reshape(-1,1), n_in=n_in)
# X_teste = df_teste_sup.drop(columns='var1(t)').values
# # Y_teste = df_teste_sup['var1(t)'].values
# Y_teste = np.copy(teste)
    
# # Treinar ELM
# ## Testando com 20 neurônios na camada escondida
# n_h = 20
# # elm = ELMRegressor(n_h)
# # elm.fit(X_treino, Y_treino)

# # Previsões
# # primeira pred
# # Y_pred = np.empty(Y_teste.shape)
# # Y_pred[0] = elm.predict(X_teste_pred[:,:n_in])
# X_teste_pred_pool = np.copy(X_teste_pred)
# # for i in range(1,Y_teste.shape[0]):
# #     X_teste_pred[:, n_in+i-1] = Y_pred[i-1] 
# #     X_teste_temp = X_teste_pred[:,i:i+n_in]
# #     Y_pred[i] = elm.predict(X_teste_temp)

# ## RMSE de teste
# # retornar para a escala normal
# Y_treino_desnorm = scaler.inverse_transform(Y_treino)
# Y_teste_desnorm = scaler.inverse_transform(Y_teste)
# # Y_pred_desnorm = scaler.inverse_transform(Y_pred)


# # Utilizar a função gera_pool
# pool_size = 100
# elm_pool = gera_pool(pool_size, n_h, X_treino, Y_treino)

# #TODO: criar diversos pool, variando a quantidade de neurônios na camada de entrada e camada escondida e pegar o melhor de cada um
# # n_h_pool = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# # elm_pool = np.empty(pool_size, ELMRegressor)
# # for n_h_i in n_h_pool:
# #     pool = np.array(gera_pool(pool_size, n_h_i, X_treino, Y_treino))
# #     pred_temp = [p.predict(X_treino) for p in pool]
# #     rmse_temp = [mean_squared_error(Y_treino, Y_pred_p, squared=True) for Y_pred_p in pred_temp]
# #     # best_p = np.argmin(rmse_temp)
# #     bests_p = np.argsort(rmse_temp)
# #     elm_pool[n_h_i - 10:n_h_i] = pool[bests_p][:10]

# # previsão do pool no treinamento
# predictions_treino_pool = [p.predict(X_treino) for p in elm_pool]
# # previsão do pool no teste
# predictions_teste_pool = pred_pool(elm_pool, n_in, Y_teste, X_teste_pred_pool)

# # scaler inverse da previsao
# predictions_treino_pool_desnorm = [scaler.inverse_transform(p) for p in predictions_treino_pool]
# predictions_teste_pool_desnorm = [scaler.inverse_transform(p) for p in predictions_teste_pool]

# # # teste de plot de todas as previsões
# # plt.plot(Y_teste_desnorm, label='Y_teste')
# # [plt.plot(t) for t in predictions_teste_pool_desnorm[:5]]
# # plt.legend()
# # plt.show()

# # predictions_pool_mean = np.mean(predictions_pool_desnorm, axis=0)
# # predictions_pool_median = np.quantile(predictions_pool_desnorm, 0.5)


# print('\nInicializar PSO')
# print('--------------------')

# Y_pred_teste_desnorm = np.array(predictions_teste_pool_desnorm).reshape(100, pool_size) # test_size, pool_size
# # Y_pred_desnorm = np.array(predictions_treino_pool_desnorm).reshape(990, pool_size) # treinamento
# Y_pred_treino = np.array(predictions_treino_pool).reshape(990, pool_size)

# def weighted_average_ensemble(p):
#     pnorm = p/p.sum()
#     res = 1/Y_pred_treino.shape[0] * (pnorm * Y_pred_treino).sum(axis=1, keepdims=True)
	
#     return res

# def forward(pesos):
#     Y_pred = weighted_average_ensemble(pesos)
#     # loss = mean_squared_error(Y_treino, Y_pred, squared=True) 
#     loss = NMSE(Y_treino, Y_pred)
    
#     return loss

# def f(x):
# 	"""
# 	Higher-level method to do the fitness in the whole swarm
# 	"""
# 	n_particles = x.shape[0]
# 	j = [forward(x[i]) for i in range(n_particles)]
# 	return np.array(j)

# # inicializar swarm
# options = {'c1': 1.49618, 'c2': 1.49618, 'w':0.7298}
# optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=pool_size, options=options)
# # Perform optimization
# cost, pos = optimizer.optimize(f, iters=100)

# # aplicar pesos aos dados de teste
# pesos_pso = np.array(pos)
# # normalizar pesos
# pesos_pso_norm = pesos_pso/pesos_pso.sum()

# # calcular a previsão do Ensemble
# Y_pred_teste = np.array(predictions_teste_pool).reshape(100, pool_size)
# Y_pred_ensemble = 1/Y_pred_teste.shape[0] * (pesos_pso_norm * Y_pred_teste).sum(axis=1, keepdims=True)

# print('Y_pred_ensemble:', Y_pred_ensemble[:5])
# print('Y_teste: ', Y_teste[:5])

# # calcular o erro
# loss = mean_squared_error(Y_teste, Y_pred_ensemble, squared=True) # rmse
# nmse_loss = NMSE(Y_teste, Y_pred_ensemble)

# print('Ensemble')
# print('RMSE: ', loss)
# print('NMSE: ', nmse_loss)

# print('-------')
# print('Média')
# print('RMSE: ', mean_squared_error(Y_teste, Y_pred_teste.mean(axis=1), squared=True))
# print('NMSE: ', NMSE(Y_teste, Y_pred_teste.mean(axis=1)))

# print('------ ')
# print('Mediana')
# print('RMSE: ', mean_squared_error(Y_teste, np.quantile(Y_pred_teste, 0.5, axis=1), squared=True))
# print('NMSE: ', NMSE(Y_teste, np.quantile(Y_pred_teste,0.5, axis=1)))


# # gráfico
# plt.plot(Y_teste_desnorm, label='Real output')
# plt.plot(scaler.inverse_transform(Y_pred_ensemble), label='Ensemble output')
# plt.plot(Y_pred_teste_desnorm.mean(axis=1), label='Média')
# plt.plot(np.quantile(Y_pred_teste_desnorm, q=0.5, axis=1), label='Mediana')
# plt.legend()
# plt.grid()
# plt.show()

#  # test_size, pool_size
# #TODO: Ordenar ELM pelo erro

# #TODO: Adicionar um modelo por vez ao ensemble. a cada entrada de modelo, otimizar com PSO


# # iniciar um laço de repetição para testar o enesemble para 1, 2, ..., N modelos
# ## Utilizar o PSO para otimizar os pesos dos modelos ELM na combinação (média ponderada)

# # Realizar previsão h+1, incorpora na série e prevê h+2, ..., até o último ponto da série