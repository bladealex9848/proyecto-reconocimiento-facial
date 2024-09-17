import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import dlib

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
    .reportview-container { background: #f0f2f6 }
    .sidebar .sidebar-content { background: #2c3e50 }
    .Widget>label { color: white; font-weight: bold; }
    .stButton>button { color: white; background-color: #3498db; border-radius: 5px; }
    .css-1aumxhk { background-color: #ffffff; border-radius: 10px; padding: 1rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    </style>
    """, unsafe_allow_html=True)

# Funciones de utilidad
@st.cache_resource
def load_models():
    faceNet = cv2.dnn.readNet("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
    emotionModel = load_model("models/modelFEC.h5")
    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    return faceNet, emotionModel, faceDetector, landmarkDetector

faceNet, emotionModel, faceDetector, landmarkDetector = load_models()

def predict_emotion(frame, faceNet, emotionModel):
    classes = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")
            
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            
            pred = emotionModel.predict(face)[0]
            label = classes[pred.argmax()]
            
            faces.append(face)
            locs.append((Xi, Yi, Xf, Yf))
            preds.append((label, pred))
    
    return locs, preds

def estimate_age_gender(face, faceDetector, landmarkDetector):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    rects = faceDetector(gray, 1)
    
    if len(rects) > 0:
        shape = landmarkDetector(gray, rects[0])
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        # Estimación simple de edad basada en proporciones faciales
        eye_distance = np.linalg.norm(shape[36] - shape[45])
        face_height = np.linalg.norm(shape[8] - shape[27])
        ratio = eye_distance / face_height
        
        if ratio > 0.25:
            age_range = "18-30"
        elif 0.24 < ratio <= 0.25:
            age_range = "30-45"
        else:
            age_range = "45+"
        
        # Estimación simple de género basada en características faciales
        jaw_width = np.linalg.norm(shape[0] - shape[16])
        forehead_width = np.linalg.norm(shape[17] - shape[26])
        
        if jaw_width > forehead_width:
            gender = "Masculino"
        else:
            gender = "Femenino"
        
        return age_range, gender
    else:
        return "Desconocido", "Desconocido"

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

def plot_emotion_chart(emotions):
    fig, ax = plt.subplots()
    emotions_df = pd.DataFrame(list(emotions.items()), columns=['Emoción', 'Confianza'])
    emotions_df = emotions_df.sort_values('Confianza', ascending=True)
    ax.barh(emotions_df['Emoción'], emotions_df['Confianza'])
    ax.set_xlabel('Confianza (%)')
    ax.set_title('Análisis Detallado de Emociones')
    st.pyplot(fig)

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
            locs, preds = predict_emotion(image, faceNet, emotionModel)
            age_range, gender = estimate_age_gender(face, faceDetector, landmarkDetector)
            
            # Dibujar bounding box y etiquetas
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            label = f"{preds[0][0]}: {preds[0][1].max()*100:.2f}%"
            cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            with col2:
                st.image(image, caption="Facial Recognition", use_column_width=True)
            
            st.subheader("Resultados del Análisis")
            
            col3, col4 = st.columns(2)
            
            with col3:
                with st.container():
                    st.markdown("##### Información Detectada")
                    st.write(f"**Emoción Predominante:** {preds[0][0]}")
                    st.write(f"**Confianza:** {preds[0][1].max()*100:.2f}%")
                    st.write(f"**Rango de Edad Estimado:** {age_range}")
                    st.write(f"**Género Estimado:** {gender}")
                    st.write(f"**ID de Participante:** #{hash(str(bbox)) % 100000:05d}")
                
                if st.button("Registrar Asistencia"):
                    st.success("Asistencia registrada exitosamente")
            
            with col4:
                emotions_dict = {e: p*100 for e, p in zip(['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido'], preds[0][1])}
                plot_emotion_chart(emotions_dict)
                
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
- Estimación de edad y género
- Registro de asistencia digital
- Generación de informes
""")

# Créditos
st.sidebar.markdown("---")
st.sidebar.subheader("Desarrollado por:")
st.sidebar.markdown("Alexander Oviedo Fadul")
st.sidebar.markdown("[GitHub](https://github.com/bladealex9848) | [Website](https://alexanderoviedofadul.dev/) | [LinkedIn](https://www.linkedin.com/in/alexander-oviedo-fadul/)")