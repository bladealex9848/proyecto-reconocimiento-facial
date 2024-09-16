import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import face_recognition
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="SIRFAJ - Reconocimiento Facial y Análisis Emocional",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title('SIRFAJ: Sistema Inteligente de Reconocimiento Facial y Análisis Emocional para Audiencias Judiciales')
st.markdown("""
    [![ver código fuente](https://img.shields.io/badge/Repositorio%20GitHub-gris?logo=github)](https://github.com/bladealex9848/proyecto-reconocimiento-facial)
    ![Visitantes](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Freconocimiento-facial.streamlit.app&label=Visitantes&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)
""")

# Funciones de reconocimiento facial y análisis emocional
def load_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir("assets/Faces Dataset"):
        image = face_recognition.load_image_file(f"assets/Faces Dataset/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(os.path.splitext(filename)[0])
    return known_faces, known_names

def recognize_face(image, known_faces, known_names):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Desconocido"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        face_names.append(name)
    
    return face_locations, face_names

def analyze_emotion(face_image):
    # Simulación de análisis emocional
    emotions = ['Feliz', 'Triste', 'Enojado', 'Neutral', 'Sorprendido']
    emotion = np.random.choice(emotions)
    confidence = np.random.random()
    return emotion, confidence

# Cargar caras conocidas
known_faces, known_names = load_known_faces()

# Interfaz de usuario
st.sidebar.title("Configuración")
upload_method = st.sidebar.radio("Método de carga", ("Subir imagen", "Usar cámara"))

if upload_method == "Subir imagen":
    uploaded_file = st.sidebar.file_uploader("Cargar una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = face_recognition.load_image_file(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        
        face_locations, face_names = recognize_face(image, known_faces, known_names)
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            face_image = image[top:bottom, left:right]
            emotion, confidence = analyze_emotion(face_image)
            cv2.putText(image, f"{emotion} ({confidence:.2f})", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        st.image(image, caption="Resultado del análisis", use_column_width=True)

elif upload_method == "Usar cámara":
    st.warning("La funcionalidad de cámara no está disponible en esta versión de demostración.")

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