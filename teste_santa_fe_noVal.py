import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils import series_to_supervised, pred_pool
from sklearn import preprocessing
from utils import NMSE
# usar o arg squared = True para calcular o RMSE
from skelm import ELMRegressor

# importando dados Santa Fe
df_sf = pd.read_csv('dados/df_santa_fe.csv')

# normalização gaussiana
scaler = preprocessing.StandardScaler().fit(df_sf['x'].values.reshape(-1,1))
df = scaler.transform(df_sf['x'].values.reshape(-1,1))

def test(serie, resultados_treinamento_dict, pool_size, n_inputs, n_hidden, scaler):
    Y_teste = serie[1000:]
    Y_teste_desnorm = scaler.inverse_transform(Y_teste)
    metricas = ['rmses', 'rmses_desnorm','rmses_media', 
                'rmses_media_desnorm','rmses_median', 'rmses_median_desnorm', 
                'predictions_pso', 'predictions_mean', 'predictions_median']

    resultados = dict.fromkeys(n_inputs)
    for (i, n_in) in enumerate(n_inputs):
        print('n_in: ', n_in)
        resultados_hidden = dict.fromkeys(n_hidden)
        X_teste_pred = serie[(1000-n_in):].T
    
        for (j, n_h) in enumerate(n_hidden):
            print('n_h: ', n_h)
            resultados_n_pool = dict.fromkeys(list(range(2, pool_size+1)))
        
            # for n_pool in range(2,pool_size+1):
            for n_pool in range(1, pool_size+1):
                print('n_pool: ', n_pool)
                resultados_metricas = dict.fromkeys(metricas)
                X_teste_pred_pool = np.copy(X_teste_pred)

                # predictions_teste_pool = pred_pool(pool[i][j][n_pool-2], n_h, Y_teste, X_teste_pred_pool)
                predictions_teste_pool = pred_pool(resultados_treinamento_dict[n_in][n_h][n_pool]['pool'][0], n_in, Y_teste, X_teste_pred_pool) #ou então str(n_pool) dependendo de como foi estruturado o dict
                predictions_teste_pool_desnorm = [scaler.inverse_transform(p) for p in predictions_teste_pool]

                Y_pred_teste = np.array(predictions_teste_pool).reshape(100, n_pool)
                Y_pred_teste_desnorm = np.array(predictions_teste_pool_desnorm).reshape(100, n_pool)
                Y_pred_ensemble = (resultados_treinamento_dict[n_in][n_h][n_pool]['pesos_pso'] * Y_pred_teste).sum(axis=1,keepdims=True)
                Y_pred_ensemble_desnorm = scaler.inverse_transform(Y_pred_ensemble)

                # calcular o RMSE
                loss = mean_squared_error(Y_teste, Y_pred_ensemble, squared=True)
                # loss_desnorm = mean_squared_error(Y_teste_desnorm, Y_pred_ensemble_desnorm, squared=True)
                loss_nmse = NMSE(Y_teste, Y_pred_ensemble)

                # resultados_metricas['predictions_pso'] = Y_pred_ensemble
                resultados_metricas['rmses'] = loss
                # resultados_metricas['rmses_desnorm'] = loss_desnorm
                resultados_metricas['rmses_desnorm'] = loss_nmse
                resultados_metricas['rmses_media'] = mean_squared_error(Y_teste, Y_pred_teste.mean(1), squared=True)
                # resultados_metricas['rmses_media_desnorm'] = mean_squared_error(Y_teste_desnorm, Y_pred_teste_desnorm.mean(1), squared=True)
                resultados_metricas['rmses_media_desnorm'] = NMSE(Y_teste_desnorm, Y_pred_teste_desnorm.mean(1))
                resultados_metricas['rmses_median'] = mean_squared_error(Y_teste, np.quantile(Y_pred_teste, 0.5, axis=1), squared=True)
                # resultados_metricas['rmses_median_desnorm'] = mean_squared_error(Y_teste_desnorm, np.quantile(Y_pred_teste_desnorm, 0.5, axis=1), squared=True)
                resultados_metricas['rmses_median_desnorm'] = NMSE(Y_teste_desnorm, np.quantile(Y_pred_teste_desnorm, 0.5, axis=1))
                resultados_metricas['predictions_pso'] = Y_pred_ensemble_desnorm
                resultados_metricas['predictions_mean'] = Y_pred_teste_desnorm.mean(1)
                resultados_metricas['predictions_median'] = np.quantile(Y_pred_teste_desnorm, 0.5, axis=1)

                resultados_n_pool[n_pool] = resultados_metricas

            resultados_hidden[n_h] = resultados_n_pool

        resultados[n_in] = resultados_hidden

    return resultados


import pickle

with open('resultados/resultados_treinamento_noVal.pickle', 'rb') as file:
    resultados_treinamento = pickle.load(file)

# n_inputs = [i for i in range(10, 210, 10)]
# n_hidden = [i for i in range(10, 160, 10)]
n_inputs = [10,20,30,40,50,100,150]
n_hidden = [20, 50, 100, 110, 150, 160, 170]

# pool = np.load('resultados/pool.npy')
# pesos_pso = np.load('resultados/pesos_pso.npy')
# erros = np.load('resultados/erros.npy')

resultados_teste = test(df, resultados_treinamento, 100, n_inputs, n_hidden, scaler)

with open('resultados/resultados_teste_noVal.pickle', 'wb') as file:
    pickle.dump(resultados_teste, file)

#TODO: ler dados de previsão gerados pelo script treinamento.py
#TODO: utilizar os ensembles gerados para calcular as previsões
#TODO: salvar resultados da melhor ELM, Melhor Ensemble, Média e Mediana do ensemble
#TODO: criar dados da paisagem (treinamento - RMSE x inputs x hidden
#TODO: criar gráfico RMSE em função do número de modelos no ensemble e mostrar PSO, Média e Mediana
#TODO: Desnormalizar e plotar gráficos e calcular o RMSE!!!
#TODO: EXPLICAR AO PROFESSOR O TEMPO DE PROCESSAMENTO SE TORNOU INVIÁVEL (MOSTRAR A CONTA DE QUANTO TEMPO DEMORARIA PARA RODAR TUDO!!!!!)
#TODO: Eles fazem 10 execcuções, teria que multiplicar todo o tempo vezes 10