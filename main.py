import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import io
import time

# Intentar importar bibliotecas opcionales
try:
    import face_recognition
    face_recognition_available = True
except ImportError:
    face_recognition_available = False
    st.warning("La biblioteca face_recognition no está disponible. Algunas funcionalidades estarán limitadas.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    tensorflow_available = True
except ImportError:
    tensorflow_available = False
    st.warning("La biblioteca TensorFlow no está disponible. El análisis emocional no estará disponible.")

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
        try:
            faceNet = cv2.dnn.readNet("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
            emotionModel = load_model("models/modelFEC.h5")
            return faceNet, emotionModel
        except Exception as e:
            st.error(f"Error al cargar los modelos: {e}")
    return None, None

def detect_faces(image):
    if face_recognition_available:
        return face_recognition.face_locations(image)
    return []

def analyze_emotion(face_image):
    if tensorflow_available:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = cv2.resize(face_image, (48, 48))
        face_image = np.expand_dims(face_image, axis=[0, -1])
        
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
    st.warning("La funcionalidad de cámara web no está disponible en esta versión de demostración.")

# Visualización con Matplotlib
st.header("Estadísticas de Emociones")

# Datos de ejemplo (reemplazar con datos reales cuando estén disponibles)
emotions = ['Feliz', 'Triste', 'Enojado', 'Neutral', 'Sorprendido']
counts = [30, 10, 5, 40, 15]

fig, ax = plt.subplots()
ax.bar(emotions, counts)
ax.set_ylabel('Cantidad')
ax.set_title('Distribución de Emociones Detectadas')
plt.xticks(rotation=45)
st.pyplot(fig)

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
for _ in range(10):  # Limitamos a 10 iteraciones para evitar bucles infinitos en Streamlit Cloud
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