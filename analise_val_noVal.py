import numpy as np
import pandas as pd
import pickle

# ler série
df_sf = pd.read_csv('dados/df_santa_fe.csv')


###################### SEM CONSIDERAR VALIDAÇÃO #######################################################

with open('resultados/resultados_treinamento_noVal.pickle', 'rb') as file:
    dados_treinamento = pickle.load(file)

# leitura dos resultados
with open('resultados/resultados_teste_noVal.pickle', 'rb') as file:
    dados = pickle.load(file)

n_inputs = [10,20,30,40,50,100,150]
n_hidden = [20, 50, 100, 110, 150, 160, 170]
res_treinamento = [[x,y] for x in n_inputs for y in n_hidden]

for i, val in enumerate(res_treinamento):
    res_treinamento[i].append(dados_treinamento[val[0]][val[1]][100]['erros'])
# criando dataframe da lista
df_res_treinamento = pd.DataFrame(res_treinamento).rename(columns={0: 'n_input', 1: 'n_hidden', 2: 'RMSE'})

res = [[x,y] for x in n_inputs for y in n_hidden]

# salvando rmse na lista
for i, val in enumerate(res):
    res[i].append(dados[val[0]][val[1]][100]['rmses'])
    # res[i].append(dados[val[0]][val[1]][100]['rmses_desnorm'])
# criando dataframe da lista
df_res = pd.DataFrame(res).rename(columns={0: 'n_input', 1: 'n_hidden', 2: 'RMSE'})

# CÁLCULO DA TAXA MÉDIA DE VARIAÇÃO DO ERRO ENTRE TREINAMENTO E TESTE
media_var_erro_tr_test = ((df_res['RMSE'] - df_res_treinamento['RMSE']) / df_res_treinamento['RMSE']).mean()
print('--------------------------------------------------------------------------------------------------------------------')
print('Taxa média de variação do erro entre treinamento e teste sem validação:', media_var_erro_tr_test)
print('--------------------------------------------------------------------------------------------------------------------')


###################### CONSIDERANDO VALIDAÇÃO #########################################################

# leitura dos resultados de treinamento
with open('resultados/resultados_treinamento.pickle', 'rb') as file:
    dados_treinamento = pickle.load(file)

# leitura dos resultados
with open('resultados/resultados_teste.pickle', 'rb') as file:
    dados = pickle.load(file)


n_inputs = [10,20,30,40,50,100,150]
n_hidden = [20, 50, 100, 110, 150, 160, 170]
res_treinamento = [[x,y] for x in n_inputs for y in n_hidden]

for i, val in enumerate(res_treinamento):
    res_treinamento[i].append(dados_treinamento[val[0]][val[1]][100]['erros'])
# criando dataframe da lista
df_res_treinamento = pd.DataFrame(res_treinamento).rename(columns={0: 'n_input', 1: 'n_hidden', 2: 'RMSE'})

res = [[x,y] for x in n_inputs for y in n_hidden]

# salvando rmse na lista
for i, val in enumerate(res):
    res[i].append(dados[val[0]][val[1]][100]['rmses'])
    # res[i].append(dados[val[0]][val[1]][100]['rmses_desnorm'])
# criando dataframe da lista
df_res = pd.DataFrame(res).rename(columns={0: 'n_input', 1: 'n_hidden', 2: 'RMSE'})

# CÁLCULO DA TAXA MÉDIA DE VARIAÇÃO DO ERRO ENTRE TREINAMENTO E TESTE
media_var_erro_tr_test_validacao = ((df_res['RMSE'] - df_res_treinamento['RMSE']) / df_res_treinamento['RMSE']).mean()
print('--------------------------------------------------------------------------------------------------------------------')
print('Taxa média de variação do erro entre treinamento e teste considerando validação:', media_var_erro_tr_test_validacao)
print('--------------------------------------------------------------------------------------------------------------------')

