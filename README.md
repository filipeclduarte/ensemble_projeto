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

Configuração da ELM:

TODO: elaborar a topologia da rede e criação do ensemble (sequencial -> melhor ELM primeiro ...)
* step size (qtd de neurônios na camada de entrada): [10, 20, ..., 90, 100] 

Lembrar de normalizar os pesos entre 0 e 1 antes de avaliar a função de aptidão.

Parâmetros do PSO:

* 100 iterações
* 100 particles
* w = 0.7298
* c1 e c2 = 1.49618

OBS: calcular Média e Mediana dos modelos ELM e comparar com o Ensemble PSO proposto 