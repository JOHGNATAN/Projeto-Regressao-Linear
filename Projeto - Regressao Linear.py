#!/usr/bin/env python
# coding: utf-8

# <h1 style='color: green; font-size: 36px; font-weight: bold;'>Data Science - Regressão Linear</h1>

# ## O Dataset e o Projeto
# <hr>
# 
# ### Fonte: https://www.kaggle.com/greenwing1985/housepricing
# 
# ### Descrição:
# <p style='font-size: 18px; line-height: 2; margin: 10px 50px; text-align: justify;'>O objetivo é criar um modelo de machine learning, utilizando a técnica de Regressão Linear, que faça previsões sobre os preços de imóveis a partir de um conjunto de características conhecidas dos imóveis.</p>
# 
# <p style='font-size: 18px; line-height: 2; margin: 10px 50px; text-align: justify;'>Vamos utilizar um dataset disponível no Kaggle que foi gerado por computador para treinamento de machine learning.</p>
# 
# 
# ### Dados:
# <ul style='font-size: 18px; line-height: 2; text-align: justify;'>
#     <li><b>precos</b> - Preços do imóveis</li>
#     <li><b>area</b> - Área do imóvel</li>
#     <li><b>garagem</b> - Número de vagas de garagem</li>
#     <li><b>banheiros</b> - Número de banheiros</li>
#     <li><b>lareira</b> - Número de lareiras</li>
#     <li><b>marmore</b> - Se o imóvel possui acabamento em mármore branco (1) ou não (0)</li>
#     <li><b>andares</b> - Se o imóvel possui mais de um andar (1) ou não (0)</li>
# </ul>

# ## Leitura dos dados
# 
# Dataset está na pasta "Dados" com o nome "HousePrices_HalfMil.csv" em usa como separador ";".

import pandas as pd 
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

dados = pd.read_csv("C:/Users/JOHGNATAN/OneDrive/Área de Trabalho/Python_Data_Science/base_de_dados_diversos/HousePrices_HalfMil.csv", 
                   sep =';')
dados


# # <font color='red' style='font-size: 30px;'>Análises Preliminares</font>
# <hr style='border: 2px solid red;'>

# ## Estatísticas descritivas


dados.describe().round(2)


# ## Matriz de correlação
# 
# <p style='font-size: 18px; line-height: 2; margin: 10px 50px; text-align: justify;'>O <b>coeficiente de correlação</b> é uma medida de associação linear entre duas variáveis e situa-se entre <b>-1</b> e <b>+1</b> sendo que <b>-1</b> indica associação negativa perfeita e <b>+1</b> indica associação positiva perfeita.</p>
# 
# 


dados.corr().round(4)


# # <font color='red' style='font-size: 30px;'>Comportamento da Variável Dependente (Y)</font>
# <hr style='border: 2px solid red;'>

# # Análises gráficas

# ![Box-Plot.png](attachment:Box-Plot.png)

# ## Box plot da variável *dependente* (X)
# 
# 

ax = sns.boxplot(dados.precos, orient='h', width=0.2)
ax.figure. set_size_inches(12,6)
ax.set_title('Preço dos Imóveis', fontsize=16)
ax.set_xlabel('$',fontsize=18)
ax


# 
# <ul style='font-size: 16px; line-height: 2; text-align: justify;'>
#     <li>Após gerar um Boxplot  da variável dependente não foi identificada a presença de quaisquer valores discrepantes ou outliers. Portanto, podemos afirmar que o dataset não possui dados discrepantes.</li>
#     <br>
#     <li>Após uma análise cuidadosa do boxplot do conjunto de dados em questão, foi identificada uma leve assimetria à direita. Isso sugere que a distribuição dos dados é ligeiramente desviada para valores mais altos.
# 
# Essa informação pode ser útil para compreender melhor a distribuição dos dados e tomar decisões mais precisas em relação à análise e interpretação dos resultados obtidos.</li>
# </ul>

# ## Investigando a variável *dependente* (Y) juntamente com outras característica
# 
# 
# 

# ### Box-plot (Preço X Garagem)


ax = sns.boxplot(x = 'garagem', y = 'precos', data = dados, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Preço X Garagem', fontsize=18)
ax.set_xlabel('Garagem', fontsize=14)
ax.set_ylabel('Preços', fontsize=14)
ax


# ### Box-plot (Preço X Banheiros)


ax = sns.boxplot(x = 'banheiros', y = 'precos', data = dados, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Preço X Banheiros', fontsize=18)
ax.set_xlabel('Banheiros', fontsize=14)
ax.set_ylabel('Preços', fontsize=14)
ax


# ### Box-plot (Preço X Lareira)


ax = sns.boxplot(x = 'lareira', y = 'precos', data = dados, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Preço X Lareira', fontsize=18)
ax.set_xlabel('Número de Lareiras', fontsize=14)
ax.set_ylabel('Preços', fontsize=14)
ax


# ### Box-plot (Preço X Acabamento em Mármore)

ax = sns.boxplot(x = 'precos', y = 'marmore', data = dados, orient='h', width=0.5)
ax.set_title('Preço X Acabamento em Mármore', fontsize=18)
ax.set_xlabel('Mármore Branco', fontsize=14)
ax.set_ylabel('Preços', fontsize=14)

ax


# <li>Após uma análise cuidadosa do boxplot do conjunto de dados em questão, foi identificada uma assimetria à direita significante. Isso significa que o acabamento em mármore é relevante, uma vez que a utilização de materiais nobres como o mármore pode valorizar consideravelmente o imóvel.</li>
# <br>
# <li>Dessa forma, é possível afirmar que a presença de acabamentos em mármore pode ser um fator determinante para aumentar o valor de venda do imóvel. dados é ligeiramente desviada para valores mais altos.
# </li>
# </ul>

# ### Box-plot (Preço X Andares)

ax = sns.boxplot(x = 'andares', y = 'precos', data = dados, orient='v', width=0.4)
ax.figure.set_size_inches(12,6)
ax.set_title('Preço X Andares', fontsize=18)
ax.set_xlabel('Andares', fontsize=14)
ax.set_ylabel('Preços', fontsize=14)
ax


# <li>imóveis com múltiplos andares tendem a ter um valor mais alto. Isso se deve ao fato de que, em geral, imóveis com mais de um andar oferecem mais espaço e conforto para os moradores, além de uma maior privacidade em relação a vizinhos e ruídos externos.</li>
# <br>
# <li>Além disso, imóveis com mais de um andar também podem apresentar uma maior flexibilidade em termos de layout e utilização de espaço, o que pode ser uma vantagem para quem busca um imóvel com necessidades específicas.</li>

# ## Distribuição de frequências da variável *dependente* (y)
# 


ax = sns.distplot(dados.precos)
ax.figure.set_size_inches(12,6)
ax.set_title('Distribuição de Frequências', fontsize=18)
ax.set_ylabel('Frequências', fontsize=18)
ax.set_xlabel('$', fontsize=18)


# ## Gráficos de dispersão entre as variáveis do dataset

# ## Plotando o pairplot fixando somente uma variável no eixo y
# 
# 


ax = sns.pairplot(dados, y_vars = 'precos', x_vars = ['area', 'garagem', 'banheiros', 'lareira', 'marmore','andares'])
ax.fig.suptitle('Dispersão entre Variáveis', fontsize=20, y = 1.2)


ax = sns.pairplot(dados, y_vars = 'precos', x_vars = ['area', 'garagem', 'banheiros', 'lareira', 'marmore','andares'],
                  kind='reg')
ax.fig.suptitle('Dispersão entre Variáveis', fontsize=20, y = 1.2)


# # <font color='red' style='font-size: 30px;'>Estimando um Modelo de Regressão Linear</font>
# <hr style='border: 2px solid red;'>

# ## Separando os dados para Teste e Treino.


y = dados.precos



X = dados[['area', 'garagem', 'banheiros', 'lareira', 'marmore','andares']]


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=2811)


# ## Instanciando a classe *LinearRegression()*


modelo = LinearRegression()


# ## Estimando o modelo linear com os dados de TREINO
# 


modelo.fit(X_train, y_train)


# ## Obtendo o coeficiente de determinação (R²) do modelo estimado com os dados de TREINO
# 


print('R² = {}'.format(modelo.score(X_train, y_train).round(2)))


# <li>O coeficiente de determinação (R²) para o modelo estimado utilizando os dados de TREINO e o resultado obtido foi de 0.64.</li>
# <br>
# <li>Esse valor indica que aproximadamente 64% da variação nos dados de TREINO pode ser explicada pelo modelo estimado. Isso sugere que o modelo tem uma boa capacidade de explicar a relação entre as variáveis
# </li>

# ## Gerando previsões para os dados de TESTE
# 
# 

y_previsto = modelo.predict(X_test)


# ## Obtendo o coeficiente de determinação (R²) para as previsões do nosso modelo
# 
# 

print('R² = {}'.format(metrics.r2_score(y_test, y_previsto).round(2)))


# <li> O coeficiente de determinação (R²) para as previsões do nosso modelo foi calculado e o resultado obtido foi de 0.67.</li>
# <br>
# <li>Esse valor indica que aproximadamente 67% da variação nas previsões do modelo pode ser explicada pelas variáveis utilizadas na modelagem. Isso sugere que o modelo tem uma boa capacidade de explicar a relação entre as variáveis e realizar previsões precisas.</li>

# # <font color='red' style='font-size: 30px;'>Obtendo Previsões Pontuais</font>
# <hr style='border: 2px solid red;'>

# ## Simulador simples
# 
# Simulador que gera estimativas de preço a partir de um conjunto de informações de um imóvel.

area = 38
garagem = 2
banheiros = 4
lareira = 4
marmore = 0
andares = 1

entrada = [[area, garagem, banheiros, lareira, marmore, andares]]

print('$ {0:.2f}'.format(modelo.predict(entrada)[0]))

