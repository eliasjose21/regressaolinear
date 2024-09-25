import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('data.csv')

colunas_indepentes_x = ["NumPassageiros", "NumBagagens", "Ar", "Km", "Tipo"]
colunas_depentes_y = ["preco"]

dados_x = df[colunas_indepentes_x]
dados_y = df[colunas_depentes_y]


modelo = LinearRegression().fit(dados_x, dados_y)

# AR = 1 sim 2 nao --- KM 0 = ilimitado --- TIPO - 1 ECONOMICO - 2 SUV - 3 COMPACTO - 4 LUXO - 5 PREMIUM   
num_pass = 5
num_baga = 1
num_ar = 1
num_km = 0
num_tipo = 1

valores_test = np.array([[num_pass, num_baga, num_ar, num_km, num_tipo]])

predicao = modelo.predict(valores_test)

print(predicao)

