#Implementado por Franklin Magalhães Ribeiro Junior

#Implementação do knn do ZERO em python!

# Para a execução do código foi utilizado a plataforma online JupyterLab (é uma IDE de python)
#Acesso por https://jupyter.org/try

# os arquivos de entrada são dadosDeTreinamento.csv e entrada.csv
#os dados de treinamento (tem 960 registros) e serão utilizados pelo algoritmo para serem #utilizados de base no cálculo de distâncias euclidianas. Já o arquivo entrada.csv contém os mais #de 44mil registros a serem classificados pelo código.

#Implementado por Franklin Magalhães Ribeiro Junior


#método da técnica kNN

def tecnicaKNN(novaTupla,tuplasTreinamento,cat):
    tamanho=len(cat)
    nDimensao=len(novaTupla)
    soma=0
    
    dist=[]
    for i in range(tamanho):
        dist.append(0)
    
    for i in range(tamanho):
        for j in range (nDimensao):
            soma= soma + ((novaTupla[j]-tuplasTreinamento[i][j]) ** 2)
        dist[i]= soma ** (1/2)
        soma=0



    kMenores = []
    for i in range(3):
        kMenores.append( [0] * 2 )

    auxiliar = []
    for i in range(tamanho):
        auxiliar.append(dist[i])

    kVizinhos=3

    for w in range(kVizinhos):
        menor=auxiliar[0]
        retirado=0

        for i in range(1,tamanho):    
            if(menor > auxiliar[i]):
                menor = auxiliar[i]
                retirado=i

        auxiliar[retirado]=9999999

        kMenores[w][0]= retirado
        kMenores[w][1]= menor



    # cat é o vetor das categorias no treinamento

    categoriaResposta = ''
    if(((cat[kMenores[0][0]])==(cat[kMenores[1][0]])) == (cat[kMenores[2][0]])):
        categoriaResposta = cat[kMenores[0][0]]
    elif (((cat[kMenores[0][0]])!=(cat[kMenores[1][0]])) != (cat[kMenores[2][0]])):
            categoriaResposta = cat[kMenores[0][0]]
    elif ((cat[kMenores[1][0]])==(cat[kMenores[2][0]])):
                categoriaResposta = cat[kMenores[1][0]]
    else:
                    categoriaResposta = cat[kMenores[0][0]]
            
    
    
    return categoriaResposta;

#fim do método


#inicio main

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

%matplotlib inline

# arquivo de treinamento
#df = pd.read_csv('dadosDeTreinamento2.csv')

#treinamento normalizado 

import pandas
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale

dados = pandas.read_csv('dadosDeTreinamento2.csv')
cols = list(dados.columns)

cols.remove('umidade')
cols.remove('categoria')

dados_amp2 = dados.copy()
dados_amp2[cols] = dados[cols].apply(minmax_scale)

cols2 = list(dados.columns)

cols2.remove('temperatura')
cols2.remove('categoria')

dados_amp2[cols2] = dados[cols2].apply(minmax_scale)

df = dados_amp2


X = np.array(df.drop('categoria',1))
y = np.array(df.categoria)

#treinou os dados

#entrar com o arquivo com os registros a serem classificados

#dfIN = pd.read_csv('entrada.csv')  # ler os dados a serem categorizados
#entrada normalizada
import pandas
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale

dados = pandas.read_csv('entrada.csv')
df2 = pd.DataFrame({"temperatura":[0, 47], 
                    "umidade":[5, 100]}) 
dados.append(df2, ignore_index = True)



cols = list(dados.columns)

cols.remove('umidade')

dados_amp = dados.copy()
dados_amp[cols] = dados[cols].apply(minmax_scale)

cols2 = list(dados.columns)

cols2.remove('temperatura')

dados_amp[cols2] = dados[cols2].apply(minmax_scale)

dfIN = dados_amp

dados= np.array(dfIN)


#classificar todos os dados com a técnica kNN (pode levar entre 2 a 3min com o tamanho 44023 dados)

categoria = []
for i in range(44023):
    tupla = dados[i]
    categoria.append(tecnicaKNN(tupla,X,y))


#gerar arquivo intermediário com as classificações

with open('intermediario.csv', 'w') as fp:
    fp.write('temperatura,umidade,categoria')
    fp.write('\n')
    for i in range(44023): 
        fp.write(str(dados[i][0])+','+str(dados[i][1])+','+categoria[i]+'\n')



#contabilizar as categorias

vetorCategorias = []
vetorCategorias.append('muito_desertico')
vetorCategorias.append('desertico')
vetorCategorias.append('critico_A')
vetorCategorias.append('quente_seco')
vetorCategorias.append('lower_fail')
vetorCategorias.append('lower_marginal')
vetorCategorias.append('lower_optimal')
vetorCategorias.append('caso_otimo')
vetorCategorias.append('upper_optimal')
vetorCategorias.append('marginal')
vetorCategorias.append('upper_marginal')
vetorCategorias.append('upper_fail')
vetorCategorias.append('frio')
vetorCategorias.append('frio e umido')
vetorCategorias.append('frio e muito umido')
vetorCategorias.append('frio e demasiado umido')

vetorContabilizarCategorias=[]
for w in range(len(vetorCategorias)):
    vetorContabilizarCategorias.append(0)

for w in range(len(vetorCategorias)):
    for i in range(44023):
        if(categoria[i]==vetorCategorias[w]):
            vetorContabilizarCategorias[w]=vetorContabilizarCategorias[w]+1



#gerar arquivo de saída

with open('saida.csv', 'w') as fp:
    fp.write('categoria,total')
    fp.write('\n')
    for w in range(len(vetorCategorias)): 
        fp.write(vetorCategorias[w]+','+str(vetorContabilizarCategorias[w])+'\n')

