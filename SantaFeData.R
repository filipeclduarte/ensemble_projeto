# li os dados:
  # SantaFe.A.rda
  # SantaFe.A.cont.rda

# treinamento e teste
treinamento <- SantaFe.A$V1
teste <- SantaFe.A.cont$V1

# juntando
df_santa_fe <- c(treinamento, teste)

# escrever dados
write.csv(df_santa_fe, 'dados/df_santa_fe.csv')

# visualizando os dados
plot(df_santa_fe, type = 'l')
