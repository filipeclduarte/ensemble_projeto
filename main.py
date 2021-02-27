import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils import series_to_supervised
# usar o arg squared = True para calcular o RMSE

# importar ELM
from ELM import * # importando sigmoid e a classe ELMRresgressor
# PSO 
# Import PySwarms
import pyswarms as ps

# importando dados Santa Fe
df_sf = pd.read_csv('dados/df_santa_fe.csv')
## treinamento as 1000 primeiras obs
df_treino = df_sf['x'].iloc[:1000].values
df_teste = df_sf['x'].iloc[1000:].values

# Estruturar os dados
## transformar o problema de série em supervised learning
### testando com 10 obs passadas
df_treino_sup = series_to_supervised(df_treino.reshape(-1,1), n_in=10)
X_treino = df_treino_sup.drop(columns='var1(t)').values
Y_treino = df_treino_sup['var1(t)'].values

df_teste_sup = series_to_supervised(df_teste.reshape(-1,1), n_in=10)
X_teste = df_teste_sup.drop(columns='var1(t)').values
Y_teste = df_teste_sup['var1(t)'].values

# Treinar ELM
## Testando com 10 neurônios na camada escondida
elm = ELMRegressor(10)
elm.fit(X_treino, Y_treino)

# Ordenar ELM pelo erro

# iniciar um laço de repetição para testar o enesemble para 1, 2, ..., N modelos
## Utilizar o PSO para otimizar os pesos dos modelos ELM na combinação (média ponderada)

# Realizar previsão h+1, incorpora na série e prevê h+2, ..., até o último ponto da série