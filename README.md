# ensemble_projeto

Projeto da disciplina de Sistemas de Múltiplos Classificadores

Alunos: 

* Filipe Coelho de Lima Duarte

* José Flávio


Replicação do artigo Extreme learning machine ensemble model for time series forecasting boosted by PSO: Application to an electric consumption problem

Método:
* ELM
* Weighted averagin Ensemble
* PSO

A função Loss é o RMSE.

Lembrar de normalizar os pesos entre 0 e 1 antes de avaliar a função de aptidão.

Parâmetros do PSO:

* 100 iterações
* 100 particles
* w = 0.7298
* c1 e c2 = 1.49618

# Executar o código: 

Dependências:

* python 3x
* NumPy
* Pandas
* Scikit-learn
* matplotlib
* statsmodel
* mpl_toolkits
* pyswarms -> https://pyswarms.readthedocs.io/en/latest/
* Scikit-ELM -> instalar conforme o tutorial https://scikit-elm.readthedocs.io/en/latest/quick_start.html

## Experimento I: Santa Fé Laser Strength

obs.: deve-se criar uma pasta chamada "resultados" no diretório principal para armazenar os resultados dos scripts abaixo.

1. executar script de treinamento no terminal:
`
python treinamento_santa_fe.py
`
2. executar script de teste no terminal:
`python teste_santa_fe.py`

3. executar análise dos resultados:
`python analise_resultados_santa_fe.py` 

## Experimento II: Consumo energia elétrica espanhola

1. executar script de treinamento no terminal:
`
python treinamento_eletric_validation.py
`
2. executar script de teste no terminal:
`python teste_eletric_validation.py`

3. executar análise dos resultados:
`python analise_resultados_eletric_validation.py` 
