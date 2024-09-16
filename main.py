import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento Facial Futurista",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para una apariencia futurista
st.markdown("""
<style>
    body {
        background-color: #1E1E1E;
        color: #00FF00;
    }
    .stButton>button {
        color: #00FF00;
        background-color: #333333;
        border: 2px solid #00FF00;
    }
    .stTextInput>div>div>input {
        color: #00FF00;
        background-color: #333333;
        border: 2px solid #00FF00;
    }
    .stSelectbox>div>div>select {
        color: #00FF00;
        background-color: #333333;
        border: 2px solid #00FF00;
    }
</style>
""", unsafe_allow_html=True)

def identificar_rostro(imagen_buscada):
    directorio_base = 'assets/Celebrity Faces Dataset/'
    resultados = []
    
    with st.spinner('Analizando rostros...'):
        for filename in os.listdir(directorio_base):
            file_path = os.path.join(directorio_base, filename)
            try:
                resultado = DeepFace.verify(imagen_buscada, file_path)
                if resultado['verified']:
                    resultados.append((filename, resultado['distance']))
            except:
                pass
    
    if resultados:
        mejor_coincidencia = min(resultados, key=lambda x: x[1])
        st.success(f"Coincidencia encontrada: {mejor_coincidencia[0]}")
        st.image(os.path.join(directorio_base, mejor_coincidencia[0]), caption="Mejor coincidencia")
        st.balloons()
    else:
        st.error("No se encontraron coincidencias")

def main():
    st.title('🤖 Sistema de Reconocimiento Facial Avanzado')
    st.subheader('Powered by DeepFace AI')

    archivo_cargado = st.file_uploader("Carga una imagen para analizar", type=['jpg', 'jpeg', 'png'])

    if archivo_cargado is not None:
        bytes_data = archivo_cargado.getvalue()
        image = Image.open(archivo_cargado)
        image_np = np.array(image)

        col1, col2, col3 = st.columns([3,1,3])

        with col1:
            st.subheader("Imagen Cargada")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Análisis")
            if st.button("Iniciar Análisis", key="analisis"):
                with st.spinner('Procesando...'):
                    try:
                        resultados = DeepFace.analyze(image_np, actions=['age', 'gender', 'emotion', 'race'])
                        st.json(resultados[0])
                    except Exception as e:
                        st.error(f"Error en el análisis: {str(e)}")

        with col3:
            st.subheader("Búsqueda de Coincidencias")
            if st.button("Buscar Coincidencias", key="buscar"):
                identificar_rostro(image_np)

    st.sidebar.title("Configuración")
    tolerancia = st.sidebar.slider("Tolerancia de coincidencia", 0.0, 1.0, 0.6, 0.01)
    st.sidebar.info("Una tolerancia más baja requiere una coincidencia más precisa.")

    # Información adicional
    st.sidebar.title("Información")
    st.sidebar.info("""
    Este sistema utiliza tecnología de inteligencia artificial para analizar y reconocer rostros.
    
    Funcionalidades:
    - Detección de edad
    - Identificación de género
    - Análisis de emociones
    - Reconocimiento étnico
    - Búsqueda de coincidencias en base de datos
    """)

if __name__ == "__main__":
    main()