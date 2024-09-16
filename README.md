# Aplicación de Reconocimiento Facial de Celebridades

![Logo de proyecto-reconocimiento-facial](https://github.com/bladealex9848/proyecto-reconocimiento-facial/blob/main/assets/logo.jpg)

## Tabla de Contenidos
1. [Descripción](#descripción)
2. [Características Principales](#características-principales)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Requisitos Previos](#requisitos-previos)
5. [Instalación](#instalación)
6. [Uso](#uso)
7. [Manual Técnico](#manual-técnico)
8. [Manual de Usuario](#manual-de-usuario)
9. [Contribución](#contribución)
10. [Registro de Cambios](#registro-de-cambios)
11. [Créditos](#créditos)
12. [Licencia](#licencia)

## Descripción

Esta aplicación de Reconocimiento Facial de Celebridades es una herramienta interactiva basada en Streamlit que permite a los usuarios cargar una imagen y identificar celebridades utilizando tecnología de reconocimiento facial. La aplicación utiliza la biblioteca `face_recognition` para detectar rostros, extraer puntos de referencia faciales y comparar el rostro cargado con un conjunto de datos de rostros de celebridades.

## Características Principales

- Carga y procesamiento de imágenes que contienen rostros
- Detección y extracción de rostros de las imágenes cargadas
- Visualización de puntos de referencia faciales en los rostros detectados
- Comparación de rostros cargados con una base de datos de rostros de celebridades
- Muestra información de la celebridad coincidente si se encuentra
- Interfaz de usuario amigable construida con Streamlit

## Estructura del Proyecto

```
proyecto-reconocimiento-facial/
│
├── main.py                            # Archivo principal de la aplicación
├── requirements.txt                   # Dependencias del proyecto
├── assets/Celebrity Faces Dataset/    # Directorio que contiene imágenes de rostros de celebridades
└── README.md                          # Documentación del proyecto (este archivo)
```

## Requisitos Previos

- Python 3.6+
- pip (gestor de paquetes de Python)
- Conexión a Internet para la instalación de dependencias

## Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/bladealex9848/proyecto-reconocimiento-facial.git
   cd proyecto-reconocimiento-facial
   ```

2. Crea un entorno virtual (opcional pero recomendado):
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows, usa `venv\Scripts\activate`
   ```

3. Instala las dependencias requeridas:
   ```
   pip install -r requirements.txt
   ```

4. Asegúrate de tener el directorio "Celebrity Faces Dataset" en la raíz de assets, que contenga imágenes de celebridades para la comparación.

## Uso

1. Ejecuta la aplicación Streamlit:
   ```
   streamlit run main.py
   ```

2. Abre tu navegador web y ve a la URL mostrada en la terminal (generalmente `http://localhost:8501`).

3. Utiliza el cargador de archivos para seleccionar y subir una imagen que contenga un rostro.

4. La aplicación procesará la imagen y mostrará:
   - La imagen cargada
   - El rostro detectado
   - Los puntos de referencia faciales
   - Los datos de codificación del rostro
   - Los resultados de la comparación con rostros de celebridades

5. Si se encuentra una celebridad coincidente, se mostrará su nombre e imagen.

## Manual Técnico

### Componentes Principales

1. **Procesamiento de Imágenes**: 
   - Utiliza OpenCV (`cv2`) para decodificar y procesar las imágenes cargadas.
   - Convierte las imágenes al espacio de color correcto (BGR a RGB).

2. **Detección de Rostros**:
   - Utiliza la biblioteca `face_recognition` para detectar rostros en la imagen cargada.
   - Extrae las ubicaciones de los rostros y los puntos de referencia faciales.

3. **Codificación de Rostros**:
   - Genera codificaciones de rostros para el rostro detectado utilizando `face_recognition`.

4. **Visualización de Puntos de Referencia Faciales**:
   - Utiliza PIL (Python Imaging Library) para dibujar los puntos de referencia faciales en el rostro detectado.

5. **Comparación de Celebridades**:
   - Compara el rostro codificado con un conjunto de datos de rostros de celebridades.
   - Utiliza `face_recognition.compare_faces` con una tolerancia de 0.6.

6. **Interfaz de Usuario**:
   - Construida con Streamlit para una aplicación web interactiva y receptiva.
   - Muestra imágenes, detalles del rostro y resultados de comparación en un diseño de varias columnas.

### Funciones Clave

- `identificarRostro(imagen_buscada)`: Compara el rostro cargado con el conjunto de datos de celebridades.

## Manual de Usuario

1. **Inicio de la Aplicación**:
   - Abre tu navegador y ve a la URL proporcionada al ejecutar la aplicación.
   - Verás una interfaz con un título "¿Qué celebridad es?" y un cargador de archivos.

2. **Carga de Imagen**:
   - Haz clic en "Elige un archivo" o arrastra y suelta una imagen en el área designada.
   - La imagen debe contener un rostro claro y visible para mejores resultados.

3. **Visualización de Resultados**:
   - Una vez cargada la imagen, verás varias secciones:
     - La imagen original cargada
     - El rostro detectado extraído de la imagen
     - Una imagen con los puntos clave del rostro marcados
     - Los datos de codificación del rostro (para usuarios técnicos)

4. **Identificación de Celebridades**:
   - La aplicación buscará automáticamente en su base de datos de celebridades.
   - Si encuentra una coincidencia, mostrará el nombre de la celebridad y su imagen.
   - Si no encuentra una coincidencia, te lo notificará.

5. **Exploración de Resultados**:
   - Puedes explorar las diferentes pestañas y secciones para ver más detalles sobre el análisis facial.

6. **Prueba con Diferentes Imágenes**:
   - Siente libre de probar con diferentes imágenes para ver cómo funciona el reconocimiento.

## Contribución

Las contribuciones a este proyecto son bienvenidas. Por favor, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/CaracteristicaIncreible`).
3. Realiza tus cambios y haz commit (`git commit -m 'Añadir alguna CaracteristicaIncreible'`).
4. Push a la rama (`git push origin feature/CaracteristicaIncreible`).
5. Abre un Pull Request.

## Registro de Cambios

### [1.0.0] - 2024-09-16
- Lanzamiento inicial de la aplicación
- Implementación de la funcionalidad básica de reconocimiento facial
- Integración con Streamlit para la interfaz de usuario

## Créditos

Desarrollado y mantenido por Alexander Oviedo Fadul, Profesional Universitario Grado 11 en el Consejo Seccional de la Judicatura de Sucre.

[GitHub](https://github.com/bladealex9848) | [Website](https://alexander.oviedo.isabellaea.com/) | [Instagram](https://www.instagram.com/alexander.oviedo.fadul) | [Twitter](https://twitter.com/alexanderofadul) | [Facebook](https://www.facebook.com/alexanderof/) | [WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!) | [LinkedIn](https://www.linkedin.com/in/alexander-oviedo-fadul-49434b9a/)

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [MIT License](https://opensource.org/licenses/MIT) para más detalles.
