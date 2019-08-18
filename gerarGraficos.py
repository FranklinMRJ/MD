#Implementado por Franklin Magalhães Ribeiro Junior
#após executar o código kNN, gerado código intermediário, saída, etc... pode plotar os gráficos

#O ideal é plotar os gráficos um a um


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

%matplotlib inline


#abrir arquivo intermediario gerado na solução proposta
qa = pd.read_csv('intermediario.csv')

#Lembre se de plotar cada gráfico por vez!!
#gerar gráfico correlograma
sb.pairplot(qa, hue = 'categoria', diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 4)



#abrir arquivo de entrada
df = pd.read_csv('entrada.csv')

#gerar gráfico da densidade da umidade na entrada
sb.distplot(df['umidade'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

#gerar gráfico da densidade da umidade na entrada
sb.distplot(df['temperatura'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

# curvas de contorno
sb.set(style="white", color_codes=True)
sb.jointplot(x=df["temperatura"], y=df["umidade"], kind='kde', color="skyblue")


