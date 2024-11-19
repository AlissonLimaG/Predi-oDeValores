import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

@st.cache
def load_data():
    return pd.read_csv("./data.csv")
def train_model():
    data = load_data()
    x = data.drop("MEDV", axis=1)
    y = data["MEDV"]
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x, y)
    return rf_regressor



data = load_data()

model = train_model()

st.title("Data app - Prevendo valoes de imóveis")

st.markdown("Este pe um data app para exibir solução de machine learning para o problema de predição de valores de imóveis de boston")

st.subheader("selecionando apenas um pequeno conjunto de atributos")

defaultcols = ["RM","PTRATIO","CHAS","MEDV"]

cols = st.multiselect("atributos", data.columns.tolist(), default=defaultcols)

st.dataframe(data[cols].head(10))

st.subheader("Distribuição de imóveis por preço")

faixa_valores = st.slider("faixa preço", float(data.MEDV.min()), 150. ,(10.0, 100.0))

dados = data[data['MEDV'].between(left=faixa_valores[0], right=faixa_valores[1])]

f = px.histogram(dados, x="MEDV", nbins=50, title='Distribuição de preços')

f.update_xaxes(title='MEDV')

f.update_yaxes(title='Total imóveis')

st.plotly_chart(f)

st.sidebar.subheader("Defina os atributos do imóvel para predição")

crim = st.sidebar.number_input("Taxa de criminalidade", value=data.CRIM.mean())

indus = st.sidebar.number_input("Proporção de hectares de negócio", value=data.INDUS.mean())

chas = st.sidebar.selectbox("Faz limite com rio?", ("sim","não"))

chas = 1 if chas == "sim" else 0

ptratio = st.sidebar.number_input("Índice de alunos para professores", value=data.PTRATIO.mean())

nox = st.sidebar.number_input("concentração de óxido nítrico", value=data.NOX.mean())

rm = st.sidebar.number_input("Número de quartos", value=1)
# lstat = st.sidebar.number_input("Proporção de pessoas com baixa renda", value=data.LSTAT.mean())

btn_predict = st.sidebar.button("Predizer preço")

if btn_predict:
    result = model.predict([[crim, indus, chas, nox, rm, ptratio]])
    st.subheader("O valor previsto para o imóvel é:")
    result = "US $" + str(round(result[0]* 10,2))
    st.write(result)