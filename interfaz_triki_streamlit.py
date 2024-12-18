import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Función para cargar modelos
@st.cache_resource
def cargar_modelos():
    return {
        "Red Densa": load_model("modelo_denso.h5"),
        "CNN": load_model("modelo_cnn.h5"),
        "LSTM": load_model("modelo_lstm.h5")
    }

# Verificar ganador
def verificar_ganador(tablero):
    combinaciones = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for combo in combinaciones:
        valores = [tablero[i] for i in combo]
        if all(v == 1 for v in valores): return "Jugador"
        if all(v == -1 for v in valores): return "IA"
    return "Empate" if 0 not in tablero else None

# Movimiento de la IA
def movimiento_ia(modelo, tablero):
    predicciones = modelo.predict(tablero.reshape(1, 9))
    movimientos_validos = np.where(tablero == 0)[0]
    return max(movimientos_validos, key=lambda x: predicciones[0][x])

# Función para entrenar modelos
def entrenar_modelo(X, y, tipo_modelo, epocas):
    if tipo_modelo == "Red Densa":
        modelo = Sequential([
            Dense(64, activation='relu', input_shape=(9,)),
            Dense(32, activation='relu'),
            Dense(9, activation='softmax')
        ])
    elif tipo_modelo == "CNN":
        modelo = Sequential([
            Reshape((3, 3, 1), input_shape=(9,)),
            Conv2D(16, (2, 2), activation='relu'),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(9, activation='softmax')
        ])
    elif tipo_modelo == "LSTM":
        modelo = Sequential([
            Reshape((9, 1), input_shape=(9,)),
            LSTM(32),
            Dense(32, activation='relu'),
            Dense(9, activation='softmax')
        ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    modelo.fit(X, y, epochs=epocas, batch_size=32, verbose=1)
    modelo.save(f"modelo_{tipo_modelo.lower().replace(' ', '_')}.h5")
    return modelo

# Cargar histórico de jugadas
def cargar_historico():
    df = pd.read_csv("jugadas_reales.csv", header=None)
    return len(df), df

# --- INTERFAZ ---
st.title("Sistema Inteligente de Triki")
menu = st.sidebar.radio("Menú", ["Jugar Contra la IA", "IA vs IA", "Reentrenar Modelos", "Ver Histórico de Jugadas"])

modelos = cargar_modelos()
tablero = np.zeros(9, dtype=int)

if menu == "Jugar Contra la IA":
    st.subheader("Jugar Contra la IA")
    modelo_seleccionado = st.selectbox("Selecciona un modelo:", list(modelos.keys()))
    modelo = modelos[modelo_seleccionado]

    for i in range(9):
        if st.button(f"Casilla {i+1}", key=f"btn{i}"):
            tablero[i] = 1
            if not verificar_ganador(tablero):
                tablero[movimiento_ia(modelo, tablero)] = -1
            st.rerun()

    st.write("### Tablero Actual")
    st.write(tablero.reshape(3, 3))
    ganador = verificar_ganador(tablero)
    if ganador: st.success(f"Resultado: {ganador}")

elif menu == "IA vs IA":
    st.subheader("Simulación IA vs IA")
    modelo1 = modelos[st.selectbox("Modelo 1", list(modelos.keys()), key="m1")]
    modelo2 = modelos[st.selectbox("Modelo 2", list(modelos.keys()), key="m2")]

    turno = 1
    while not verificar_ganador(tablero):
        tablero[movimiento_ia(modelo1 if turno == 1 else modelo2, tablero)] = turno
        turno = -turno
    st.write("### Resultado Final")
    st.write(tablero.reshape(3, 3))
    st.success(f"Resultado: {verificar_ganador(tablero)}")

elif menu == "Reentrenar Modelos":
    st.subheader("Reentrenar Modelos")
    epocas = st.slider("Número de épocas:", 10, 100, 50)
    tipo_modelo = st.selectbox("Selecciona el modelo a reentrenar:", ["Red Densa", "CNN", "LSTM"])

    st.write("Cargando datos...")
    datos = pd.read_csv("jugadas_reales.csv", header=None).values
    X, y = datos[:, :9], pd.get_dummies(datos[:, 9]).values

    if st.button("Entrenar Modelo"):
        modelo_entrenado = entrenar_modelo(X, y, tipo_modelo, epocas)
        st.success(f"Modelo {tipo_modelo} reentrenado y guardado.")

elif menu == "Ver Histórico de Jugadas":
    st.subheader("Histórico de Jugadas")
    total, df = cargar_historico()
    st.write(f"Total de jugadas registradas: {total}")
    st.dataframe(df.tail(10))
