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
df_sf = pd.read_csv('dados/df.csv')

# normalização gaussiana
scaler = preprocessing.StandardScaler().fit(df_sf['value'].values.reshape(-1,1))
df = scaler.transform(df_sf['value'].values.reshape(-1,1))

# n_inputs = [i for i in range(10, 210, 10)]
# n_hidden = [i for i in range(10, 160, 10)]
n_inputs = [30,40,50,60,90]
n_hidden = [100, 150, 160, 140]

pool_size = 100

treino = df[:5000]
teste = df[5000:24203]
Y_teste = np.copy(teste)

def treinamento(serie_treino, n_inputs, n_hidden, pool_size):
    # pool = np.empty(shape=(len(n_inputs), pool_size), dtype=ELMRegressor)
    # pesos_pso = []
    # erros = []

    resultados = dict.fromkeys(n_inputs)
    # metricas = ['pool', 'pesos_pso', 'erros']
    metricas = ['pool', 'pesos_pso', 'erros']

    for (_, n_in) in enumerate(n_inputs):
        print('n_in:', n_in)
        df_treino_sup = series_to_supervised(serie_treino.reshape(-1,1), n_in=n_in)
        X_treino = df_treino_sup.drop(columns='var1(t)').values
        Y_treino = df_treino_sup['var1(t)'].values

        len_treino = int(X_treino.shape[0] * 0.8)

        resultados_n_h = dict.fromkeys(n_hidden)
        X_treino1 = X_treino[:len_treino]
        # X_val = X_treino[len_treino:]

        # Y_treino1 = Y_treino[:len_treino]
        Y_val = Y_treino[len_treino:]

        X_val_pred = serie_treino[(len_treino-n_in):].T


        for (j, n_h) in enumerate(n_hidden):
            pool = np.empty(shape=(1, pool_size), dtype=ELMRegressor)
            print('n_h:', n_h)
            
            pool_temp = np.array(gera_pool(pool_size, n_h, X_treino, Y_treino))
            # predictions_treino_pool = [p.predict(X_treino) for p in pool_temp]
            predictions_treino_pool = [p.predict(X_treino1) for p in pool_temp]
            
            X_val_pred_pool = np.copy(X_val_pred)

            predictions_val_pool = pred_pool(pool_temp, n_in, Y_val, X_val_pred_pool)

            # Y_pred_treino = np.array(predictions_treino_pool).reshape(-1, pool_size)
            Y_pred_val = np.array(predictions_val_pool).reshape(-1, pool_size)
            # rmse_temp = [mean_squared_error(Y_treino, Y_pred_p, squared=True) for Y_pred_p in predictions_treino_pool]
            # rmse_temp = [mean_squared_error(Y_treino1, Y_pred_p, squared=True) for Y_pred_p in predictions_treino_pool]
            rmse_temp = [mean_squared_error(Y_val, Y_pred_p, squared=True) for Y_pred_p in predictions_val_pool]

            bests_p = np.argsort(rmse_temp)
            pool[:, :] = pool_temp[bests_p]
            # Y_pred_treino = Y_pred_treino[:,bests_p] # ordenando as previsões tbm
            Y_pred_val = Y_pred_val[:, bests_p]

            resultados_n_pool = dict.fromkeys(list(range(2, pool_size+1)))

            # for n_pool in range(2,pool_size+1):
            for n_pool in range(1, pool_size+1):
                print('n_pool:', n_pool, 'n_in:', n_in, 'n_h:', n_h)
                resultados_metricas = dict.fromkeys(metricas)
                if n_pool == 1:
                    resultados_metricas['pool'] = pool[:, :n_pool]
                    resultados_metricas['pesos_pso'] = np.array([1.0])
                    # resultados_metricas['erros'] = NMSE(Y_treino, Y_pred_treino[:, :n_pool])
                    # resultados_metricas['erros'] = NMSE(Y_treino1, Y_pred_treino[:, :n_pool])
                    resultados_metricas['erros'] = NMSE(Y_val, Y_pred_val[:,:n_pool])
                    resultados_n_pool[n_pool] = resultados_metricas

                else:

                    def weighted_average_ensemble(p):
                        pnorm = np.nan_to_num(np.exp(p)/np.exp(p).sum()) # testando com norm exp
                        # pnorm = np.nan_to_num((p - p.min())/(p - p.min()).sum())
                        # res = 1/Y_pred_treino[:, :n_pool].shape[0] * (pnorm * Y_pred_treino[:, :n_pool]).sum(axis=1, keepdims=True)
                        # res = (pnorm * Y_pred_treino[:, :n_pool]).sum(axis=1, keepdims=True)
                        res = (pnorm * Y_pred_val[:, :n_pool]).sum(axis=1, keepdims=True)
                        return res

                    def forward(pesos):
                        Y_pred = weighted_average_ensemble(pesos)
                        # loss = NMSE(Y_treino, Y_pred)
                        loss = NMSE(Y_val, Y_pred)
                        return loss

                    def f(x):
                        n_particles = x.shape[0]
                        j = [forward(x[i]) for i in range(n_particles)]
                        return np.array(j)

                    # testando incrementalmente 
                    options = {'c1': 1.49618, 'c2': 1.49618, 'w':0.7298}
            
                    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=n_pool, options=options)
                    cost_temp, pos = optimizer.optimize(f, iters=100)
                    
                    pesos_pso_temp = np.array(pos)
                    pesos_pso_norm = np.nan_to_num(np.exp(pesos_pso_temp)/np.exp(pesos_pso_temp).sum())

                    resultados_metricas['pool'] = pool[:, :n_pool]
                    resultados_metricas['pesos_pso'] = pesos_pso_norm
                    resultados_metricas['erros'] = cost_temp
                    # salvar as metricas
                    resultados_n_pool[n_pool] = resultados_metricas

            resultados_n_h[n_h] = resultados_n_pool
        
        resultados[n_in] = resultados_n_h

    return resultados

resultados_dict = treinamento(treino, n_inputs, n_hidden, pool_size = 100)

# np.save('resultados/pool.npy', pool)
# np.save('resultados/pesos_pso.npy', pesos_pso)
# np.save('resultados/erros.npy', erros)

import pickle

with open('resultados/resultados_treinamento_validation.pickle', 'wb') as file:
    pickle.dump(resultados_dict, file)

