import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import matplotlib.pyplot as plt

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
    faceNet = cv2.dnn.readNet("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
    emotionModel = load_model("models/modelFEC.h5")
    return faceNet, emotionModel

faceNet, emotionModel = load_models()

# Función para predecir la emoción
def predict_emotion(frame, faceNet, emotionModel):
    classes = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            
            pred = emotionModel.predict(face)[0]
            label = classes[pred.argmax()]
            
            faces.append(face)
            locs.append((Xi, Yi, Xf, Yf))
            preds.append((label, pred))
    
    return locs, preds

# Interfaz de usuario
uploaded_file = st.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    locs, preds = predict_emotion(image, faceNet, emotionModel)
    
    for (box, (label, pred)) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        cv2.rectangle(image, (Xi, Yi), (Xf, Yf), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {pred.max():.2f}", (Xi, Yi - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    st.image(image, caption="Emociones detectadas", use_column_width=True)
    
    # Gráfico de barras para las emociones
    if preds:
        fig, ax = plt.subplots()
        emotions = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']
        values = preds[0][1]
        ax.bar(emotions, values)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

# Información del sistema
st.sidebar.title("Información del Sistema")
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