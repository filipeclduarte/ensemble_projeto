import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.graphics.tsaplots import plot_acf
import pickle

# leitura dos resultados de treinamento
with open('resultados/resultados_treinamento_validation.pickle', 'rb') as file:
    dados_treinamento = pickle.load(file)

# leitura dos resultados
with open('resultados/resultados_teste_validation.pickle', 'rb') as file:
    dados = pickle.load(file)

# ler série
df_sf = pd.read_csv('dados/df.csv')

# treinamento
treinamento = df_sf['value'][:5000]
# teste
teste = df_sf['value'][5000:6000]

#plot
# plt.plot(treinamento, label='treinamento')
# plt.plot(teste, label='teste')
# plt.legend()
# plt.grid()
# plt.savefig('resultados/serie_santa_fe.png')
# plt.show()

# # plot de autocorrelacao
# plot_acf(treinamento)
# plt.savefig('resultados/acf_santa_fe.png')
# plt.show()

## 3D superfície de treinamento com pool size = 100
n_inputs = [30,40,50,60,90]
n_hidden = [100, 150, 160, 140]
res_treinamento = [[x,y] for x in n_inputs for y in n_hidden]

## dataframe
for i, val in enumerate(res_treinamento):
    res_treinamento[i].append(dados_treinamento[val[0]][val[1]][100]['erros'])
# criando dataframe da lista
df_res_treinamento = pd.DataFrame(res_treinamento).rename(columns={0: 'n_input', 1: 'n_hidden', 2: 'RMSE'})

# graph
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_trisurf(df_res_treinamento.n_hidden, df_res_treinamento.n_input, df_res_treinamento.RMSE, cmap=cm.jet, linewidth=0.2)
ax.set_ylabel('Qtd. neurônios na camada de entrada')
ax.set_xlabel('Qtd. neurônios na camada de escondida')
plt.savefig('resultados/rmse_treinanmento_eletric.png')
# plt.savefig('resultados/rmse_validacao_santa_fe.png')
plt.show()


# 3D superficie com pool_size = 100
# n_inputs = [10,20,30,40,50,100,150]
# n_hidden = [20, 50, 100, 110, 150, 160, 170]
res = [[x,y] for x in n_inputs for y in n_hidden]

# salvando rmse na lista
for i, val in enumerate(res):
    res[i].append(dados[val[0]][val[1]][100]['rmses'])
    # res[i].append(dados[val[0]][val[1]][100]['rmses_desnorm'])
# criando dataframe da lista
df_res = pd.DataFrame(res).rename(columns={0: 'n_input', 1: 'n_hidden', 2: 'RMSE'})


# graph
fig = plt.figure()
ax = Axes3D(fig)
plt.title('RMSE vs Neurônios camada escondida vs Neurônios camada de entrada')
ax.plot_trisurf(df_res.n_hidden, df_res.n_input, df_res.RMSE, cmap=cm.jet, linewidth=0.2)
ax.set_ylabel('Qtd. neurônios na camada de entrada')
ax.set_xlabel('Qtd. neurônios na camada de escondida')
plt.savefig('resultados/rmse_eletric.png')
plt.show()

# analisar configurações
print('Melhores configurações 100 modelos')
print(df_res.sort_values('RMSE'))
df_res.sort_values('RMSE').to_csv('resultados/df_res_eletric.csv')

# melhor topologia individual
res_ind = [[x,y] for x in n_inputs for y in n_hidden]
for i, val in enumerate(res_ind):
    # res_ind[i].append(dados[val[0]][val[1]][1]['rmses'])
    res_ind[i].append(dados[val[0]][val[1]][1]['rmses_desnorm'])

# analisar configurações individual
df_res_ind = pd.DataFrame(res_ind).rename(columns={0: 'n_input', 1: 'n_hidden', 2: 'RMSE'})
print('Best model individual')
print(df_res_ind.sort_values('RMSE'))

# salvar resultados
df_res_ind.sort_values('RMSE').to_csv('resultados/df_res_ind_eletric.csv')

# gráfico
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('RMSE vs Neurônios camada escondida vs Neurônios camada de entrada')
ax.plot_trisurf(df_res_ind.n_hidden, df_res_ind.n_input, df_res_ind.RMSE, cmap=cm.jet, linewidth=0.2)
ax.set_ylabel('Qtd. neurônios na camada de entrada')
ax.set_xlabel('Qtd. neurônios na camada de escondida')
plt.savefig('resultados/rmse_best_model_ind_eletric.png')
plt.show()


# gráfico number models vs rmse PSO, Median e Mean
# pegar as melhores configurações
res_pool_size = [[i] for i in range(1,101)]
# popular
for i,_ in enumerate(res_pool_size):
    res_pool_size[i].append(dados[90][140][i+1]['rmses'])
    res_pool_size[i].append(dados[90][140][i+1]['rmses_media'])
    res_pool_size[i].append(dados[90][140][i+1]['rmses_median'])
    # res_pool_size[i].append(dados[30][150][i+1]['rmses_desnorm'])
    # res_pool_size[i].append(dados[30][150][i+1]['rmses_media_desnorm'])
    # res_pool_size[i].append(dados[30][150][i+1]['rmses_median_desnorm'])
# analisar tabela
df_res_pool_size = pd.DataFrame(res_pool_size).rename(columns={0: 'pool_size', 1: 'PSO', 2: 'Média', 3: 'Mediana'})
print('RMSE melhor configuração pool size')
print(df_res_pool_size)
df_res_pool_size.to_csv('resultados/df_res_pool_size_eletric.csv')

# graph number of models in ensemble
df_res_pool_size[['PSO', 'Média', 'Mediana']].plot()
plt.grid()
plt.xlabel('Quantidade de modelos no ensemble')
plt.ylabel('RMSE')
plt.savefig('resultados/RMSE_pool_size_eletric.png')
plt.show()

## Gráfico de previsao do melhor modelo, melhor ensemble, média e mediana
res_pred = {'Y_teste': teste.values, 'Best ELM': dados[90][140][1]['predictions_pso'].reshape(-1),
            'Ensemble': dados[90][140][100]['predictions_pso'].reshape(-1), 'Média': dados[90][140][100]['predictions_mean'],
            'Mediana': dados[90][140][100]['predictions_median']}

df_res_pred = pd.DataFrame(res_pred)
pred = df_res_pred[:100]

print('Previsões')
print(df_res_pred)
pred.plot()
plt.ylabel('Laser Strength')
plt.xlabel('t')
plt.savefig('resultados/previsoes_eletric.png')
plt.show()

# salvar previsoes
df_res_pred.to_csv('resultados/previsoes_eletric.csv')