import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils import series_to_supervised
from sklearn import preprocessing
# usar o arg squared = True para calcular o RMSE

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

## treinamento as 1000 primeiras obs
treino = df[:1000]
teste = df[1000:]

# Estruturar os dados
## transformar o problema de série em supervised learning
### testando com 10 obs passadas
df_treino_sup = series_to_supervised(treino.reshape(-1,1), n_in=10)
X_treino = df_treino_sup.drop(columns='var1(t)').values
Y_treino = df_treino_sup['var1(t)'].values

df_teste_sup = series_to_supervised(teste.reshape(-1,1), n_in=10)
X_teste = df_teste_sup.drop(columns='var1(t)').values
Y_teste = df_teste_sup['var1(t)'].values

# Treinar ELM
## Testando com 10 neurônios na camada escondida
elm = ELMRegressor(10)
elm.fit(X_treino, Y_treino)

## pred
Y_pred = elm.predict(X_teste)
## RMSE de teste
RMSE = mean_squared_error(Y_teste, Y_pred, squared = True)
print(f'RMSE = {RMSE}')

## Plotar 
plt.plot(Y_teste, label='Real')
plt.plot(Y_pred, label='Prediction')
plt.legend()
plt.title('Teste')
plt.show()

#TODO: Gerar os modelos ELM

#TODO: Ordenar ELM pelo erro

#TODO: Adicionar um modelo por vez ao ensemble. a cada entrada de modelo, otimizar com PSO


# iniciar um laço de repetição para testar o enesemble para 1, 2, ..., N modelos
## Utilizar o PSO para otimizar os pesos dos modelos ELM na combinação (média ponderada)

# Realizar previsão h+1, incorpora na série e prevê h+2, ..., até o último ponto da série