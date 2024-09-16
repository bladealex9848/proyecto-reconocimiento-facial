import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from ipyvizzu import Config, Data, Style
from ipyvizzustory import Story, Slide, Step
import os
import io
import time

# Intentar importar bibliotecas opcionales
try:
    import face_recognition
    face_recognition_available = True
except ImportError:
    face_recognition_available = False

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    tensorflow_available = True
except ImportError:
    tensorflow_available = False

# Configuración de la página
st.set_page_config(
    page_title="SIRFAJ - Reconocimiento Facial y Análisis Emocional",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    </style>
    """, unsafe_allow_html=True)

# Funciones de utilidad
@st.cache_resource
def load_models():
    if face_recognition_available and tensorflow_available:
        faceNet = cv2.dnn.readNet("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
        emotionModel = load_model("models/modelFEC.h5")
        return faceNet, emotionModel
    return None, None

def detect_faces(image):
    if face_recognition_available:
        return face_recognition.face_locations(image)
    return []

def analyze_emotion(face_image):
    if tensorflow_available:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (48, 48))
        face_image = img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        
        emotion_labels = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']
        emotion_probs = emotionModel.predict(face_image)[0]
        emotion = emotion_labels[np.argmax(emotion_probs)]
        return emotion, max(emotion_probs)
    return "No disponible", 0

# Carga de modelos
faceNet, emotionModel = load_models()

# Interfaz de usuario principal
st.title('SIRFAJ: Sistema Inteligente de Reconocimiento Facial y Análisis Emocional')

# Selector de modo
mode = st.sidebar.selectbox("Seleccione el modo", ["Imagen", "Cámara Web"])

if mode == "Imagen":
    uploaded_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        
        # Procesar imagen
        img_array = np.array(image)
        faces = detect_faces(img_array)
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detección Facial")
            for (top, right, bottom, left) in faces:
                cv2.rectangle(img_array, (left, top), (right, bottom), (0, 255, 0), 2)
            st.image(img_array, channels="RGB")
        
        with col2:
            st.subheader("Análisis Emocional")
            for (top, right, bottom, left) in faces:
                face_image = img_array[top:bottom, left:right]
                emotion, confidence = analyze_emotion(face_image)
                st.write(f"Emoción: {emotion} (Confianza: {confidence:.2f})")

elif mode == "Cámara Web":
    st.warning("Asegúrese de permitir el acceso a la cámara web.")
    
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara web. Por favor, verifique la conexión.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error al capturar el frame.")
                break
            
            faces = detect_faces(frame)
            
            for (top, right, bottom, left) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                face_image = frame[top:bottom, left:right]
                emotion, confidence = analyze_emotion(face_image)
                cv2.putText(frame, f"{emotion}: {confidence:.2f}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            FRAME_WINDOW.image(frame, channels="BGR")
            
            if st.button('Detener'):
                break
        
        cap.release()

# Visualización con Vizzu
st.header("Estadísticas de Emociones")

# Datos de ejemplo (reemplazar con datos reales cuando estén disponibles)
data = Data()
data.add_df(pd.DataFrame({
    "Emoción": ["Feliz", "Triste", "Enojado", "Neutral", "Sorprendido"],
    "Cantidad": [30, 10, 5, 40, 15]
}))

story = Story(data)
story.set_size(800, 480)

story.add_slide(
    Slide(
        Step(
            Config({
                "x": "Emoción",
                "y": "Cantidad",
                "title": "Distribución de Emociones Detectadas",
                "color": "Emoción"
            }),
            Style({
                "plot": {
                    "yAxis": {"label": {"numberScale": "shortScaleSymbolUS"}},
                    "xAxis": {"label": {"numberScale": "shortScaleSymbolUS"}},
                    "marker": {
                        "colorPalette": "#3498db #e74c3c #2ecc71 #f1c40f #9b59b6"
                    }
                }
            })
        )
    )
)

story.play()

# Métricas en tiempo real
st.header("Métricas en Tiempo Real")
col1, col2, col3 = st.columns(3)

with col1:
    faces_detected = st.empty()
with col2:
    emotions_analyzed = st.empty()
with col3:
    processing_time = st.empty()

# Simulación de métricas en tiempo real (reemplazar con datos reales)
while True:
    faces_detected.metric("Rostros Detectados", np.random.randint(1, 10))
    emotions_analyzed.metric("Emociones Analizadas", np.random.randint(1, 10))
    processing_time.metric("Tiempo de Procesamiento", f"{np.random.rand():.2f} s")
    time.sleep(1)

# Información adicional
st.sidebar.markdown("---")
st.sidebar.subheader("Acerca de SIRFAJ")
st.sidebar.info("""
Este sistema utiliza tecnología de inteligencia artificial para analizar y reconocer rostros en tiempo real.

Funcionalidades:
- Reconocimiento facial
- Análisis emocional
- Registro de asistencia digital
- Generación de informes
""")

# Créditos
st.sidebar.markdown("---")
st.sidebar.subheader("Desarrollado por:")
st.sidebar.markdown("Alexander Oviedo Fadul")
st.sidebar.markdown("[GitHub](https://github.com/bladealex9848) | [Website](https://alexanderoviedofadul.dev/) | [LinkedIn](https://www.linkedin.com/in/alexander-oviedo-fadul/)")