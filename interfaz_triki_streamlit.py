import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelos pre-entrenados
modelo_denso = load_model("modelo_denso.h5")
modelo_cnn = load_model("modelo_cnn.h5")
modelo_lstm = load_model("modelo_lstm.h5")

# Verificar ganador en el tablero
def verificar_ganador(tablero):
    combinaciones = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Filas
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columnas
        [0, 4, 8], [2, 4, 6]              # Diagonales
    ]
    for combo in combinaciones:
        valores = [tablero[i] for i in combo]
        if all(v == 1 for v in valores):
            return "¡Ganaste tú!"
        if all(v == -1 for v in valores):
            return "¡La IA ganó!"
    return "Empate" if 0 not in tablero else None

# Movimiento de la IA usando el modelo seleccionado
def movimiento_ia(modelo, tablero):
    predicciones = modelo.predict(tablero.reshape(1, 9))
    movimientos_validos = np.where(tablero == 0)[0]
    mejor_movimiento = max(movimientos_validos, key=lambda x: predicciones[0][x])
    return mejor_movimiento

# Función para renderizar el tablero
def renderizar_tablero(tablero):
    cols = st.columns(3)
    for i in range(9):
        with cols[i % 3]:
            st.button("X" if tablero[i] == 1 else "O" if tablero[i] == -1 else " ", key=f"btn{i}", disabled=True)

# Inicialización del tablero
if "tablero" not in st.session_state:
    st.session_state.tablero = np.zeros(9, dtype=int)
    st.session_state.turno_jugador = True

# Título y descripción
st.title("Triki (Tres en Raya) - Juega contra la IA")
st.write("Selecciona un modelo para jugar contra la IA.")

# Selección del modelo
modelo_seleccionado = st.selectbox("Elige un modelo de IA:", ["Red Densa", "CNN", "LSTM"])
modelo = {"Red Densa": modelo_denso, "CNN": modelo_cnn, "LSTM": modelo_lstm}[modelo_seleccionado]

# Juego
st.write("## Tablero Actual")
renderizar_tablero(st.session_state.tablero)

# Turno del jugador
if st.session_state.turno_jugador:
    st.write("**Tu turno: selecciona una casilla**")
    for i in range(9):
        if st.session_state.tablero[i] == 0:
            if st.button(f"Casilla {i+1}", key=f"btn_player{i}"):
                st.session_state.tablero[i] = 1
                st.session_state.turno_jugador = False
                st.rerun()

# Turno de la IA
if not st.session_state.turno_jugador:
    movimiento = movimiento_ia(modelo, st.session_state.tablero)
    st.session_state.tablero[movimiento] = -1
    st.session_state.turno_jugador = True
    st.rerun()

# Verificar si hay ganador
resultado = verificar_ganador(st.session_state.tablero)
if resultado:
    st.success(resultado)
    if st.button("Reiniciar Juego"):
        st.session_state.tablero = np.zeros(9, dtype=int)
        st.session_state.turno_jugador = True
        st.rerun()
