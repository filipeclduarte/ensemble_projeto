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

"The procedure consists of a batch training of ELMs within a
range defined by the input vector ð Þ XM1 and the range for the
number of hidden nodes. A total of 10 ELMs are created for a specific combination, giving the performance results a more reliable statistical base. The step size in the range is also 10, i.e, the number of
inputs within a range between 10 and 100 will be
[10; 20; ... ; 90; 100] . The performance of each topology leads to a
ranking that contains the best 100 ELMs and is used to build the
ensemble. An aspect that remains in the topology along the experiments is the activation function of the hidden and output layer
nodes, which is set to Sigmoid and Linear, respectively.
Once the ELMs are tuned, the ensemble building process is carried out following the next steps:
1. The ELMs are sorted by their prediction error in ascending
order.
2. The first model in the sorted list is added to the ensemble with a
weight of 1.
3. The following model in the list is added to the ensemble, and
the PSO is called in order to compute the best weights.
4. Repeat step 3 until the maximum number of experts in the EM
is reached." 



Lembrar de normalizar os pesos entre 0 e 1 antes de avaliar a função de aptidão.

Parâmetros do PSO:

* 100 iterações
* 100 particles
* w = 0.7298
* c1 e c2 = 1.49618

OBS: calcular Média e Mediana dos modelos ELM e comparar com o Ensemble PSO proposto 