import pickle
import pandas as pd
import matplotlib.pyplot as plt
# ler dados
with open('dados/2014.pkl', 'rb') as file:
    dados_2014 = pickle.load(file)

with open('dados/2015.pkl', 'rb') as file:
    dados_2015 = pickle.load(file)

with open('dados/2016.pkl', 'rb') as file:
    dados_2016 = pickle.load(file)

# visualizar dados

df = pd.DataFrame({'datetime':[], 'value':[]})
value = []
datetime = []

dados_2014_dict = dict.fromkeys(dados_2014.keys())
for mes in dados_2014_dict:
    dias_2014 = dict.fromkeys(dados_2014[mes].keys())
    for dia in dias_2014:       
        for hora in range(len(dados_2014[mes][dia]['included'][0]['attributes']['values'])):
            value.append(dados_2014[mes][dia]['included'][0]['attributes']['values'][hora]['value'])
            datetime.append(dados_2014[mes][dia]['included'][0]['attributes']['values'][hora]['datetime'])

dados_2015_dict = dict.fromkeys(dados_2015.keys())
for mes in dados_2015_dict:
    dias_2015 = dict.fromkeys(dados_2015[mes].keys())
    for dia in dias_2015:       
        for hora in range(len(dados_2015[mes][dia]['included'][0]['attributes']['values'])):
            value.append(dados_2015[mes][dia]['included'][0]['attributes']['values'][hora]['value'])
            datetime.append(dados_2015[mes][dia]['included'][0]['attributes']['values'][hora]['datetime'])

dados_2016_dict = dict.fromkeys(dados_2016.keys())
for mes in dados_2016_dict:
    dias_2016 = dict.fromkeys(dados_2016[mes].keys())
    for dia in dias_2016:       
        for hora in range(len(dados_2016[mes][dia]['included'][0]['attributes']['values'])):
            value.append(dados_2016[mes][dia]['included'][0]['attributes']['values'][hora]['value'])
            datetime.append(dados_2016[mes][dia]['included'][0]['attributes']['values'][hora]['datetime'])

# salvar listas em df
df['datetime'] = datetime
df['value'] = value

# salvar .csv
df.to_csv('dados/df.csv')