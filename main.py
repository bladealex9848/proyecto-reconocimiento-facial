import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import matplotlib.pyplot as plt
from PIL import Image
import random

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
        background: #2c3e50
    }
    .Widget>label {
        color: white;
        font-weight: bold;
    }
    .stButton>button {
        color: white;
        background-color: #3498db;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Funciones de utilidad
@st.cache_resource
def load_models():
    faceNet = cv2.dnn.readNet("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
    emotionModel = load_model("models/modelFEC.h5")
    return faceNet, emotionModel

faceNet, emotionModel = load_models()

def predict_emotion(face):
    emotions = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']
    face = cv2.resize(face, (48, 48))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    preds = emotionModel.predict(face)[0]
    emotion = emotions[preds.argmax()]
    return emotion, preds.max()

def estimate_age_gender(face):
    # Simulación de estimación de edad y género
    age = random.randint(20, 60)
    gender = random.choice(["Masculino", "Femenino"])
    return age, gender

def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            return face, (startX, startY, endX, endY)
    
    return None, None

# Interfaz de usuario
st.sidebar.title("SIRFAJ Dashboard")
menu = st.sidebar.selectbox("Seleccione una función", 
                            ["Reconocimiento Facial", "Análisis Emocional", "Registro de Asistencia", "Generación de Informes", "Alertas de Seguridad"])

if menu == "Reconocimiento Facial" or menu == "Análisis Emocional":
    st.title("Reconocimiento Facial y Análisis Emocional")
    
    uploaded_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image.convert('RGB'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Imagen Original", use_column_width=True)
        
        face, bbox = detect_face(image)
        
        if face is not None:
            emotion, confidence = predict_emotion(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            age, gender = estimate_age_gender(face)
            
            # Dibujar bounding box y etiquetas
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            with col2:
                st.image(image, caption="Facial Recognition", use_column_width=True)
            
            st.subheader("Resultados del Análisis")
            st.write(f"Emoción Detectada: {emotion}")
            st.write(f"Confianza: {confidence:.2f}")
            st.write(f"Edad Estimada: {age} años")
            st.write(f"Género: {gender}")
            st.write(f"ID de Participante: #{random.randint(10000, 99999)}")
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("Registrar Asistencia"):
                    st.success("Asistencia registrada exitosamente")
            with col4:
                if st.button("Generar Alerta"):
                    st.warning("Alerta generada. Personal de seguridad notificado.")
        else:
            st.error("No se detectó ningún rostro en la imagen. Por favor, intente con otra imagen.")

elif menu == "Registro de Asistencia":
    st.title("Registro de Asistencia")
    st.info("Funcionalidad en desarrollo. Estará disponible próximamente.")

elif menu == "Generación de Informes":
    st.title("Generación de Informes")
    st.info("Funcionalidad en desarrollo. Estará disponible próximamente.")

elif menu == "Alertas de Seguridad":
    st.title("Alertas de Seguridad")
    st.info("Funcionalidad en desarrollo. Estará disponible próximamente.")

# Información del sistema
st.sidebar.markdown("---")
st.sidebar.subheader("Información del Sistema")
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