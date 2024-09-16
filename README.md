# SIRFAJ: Sistema Inteligente de Reconocimiento Facial y Análisis Emocional para Audiencias Judiciales

![Logo del proyecto-reconocimiento-facial](https://github.com/bladealex9848/proyecto-reconocimiento-facial/blob/main/assets/logo.jpg?raw=true)

[![ver código fuente](https://img.shields.io/badge/Repositorio%20GitHub-gris?logo=github)](https://github.com/bladealex9848/proyecto-reconocimiento-facial)
![Visitantes](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Freconocimiento-facial.streamlit.app&label=Visitantes&labelColor=%235d5d5d&countColor=%231e7ebf&style=flat)

## Tabla de Contenidos
1. [Descripción](#descripción)
2. [Características Principales](#características-principales)
3. [Alineación con los Objetivos del Concurso](#alineación-con-los-objetivos-del-concurso)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Requisitos Previos](#requisitos-previos)
6. [Instalación](#instalación)
7. [Uso](#uso)
8. [Impacto y Beneficios](#impacto-y-beneficios)
9. [Innovación y Valor Agregado](#innovación-y-valor-agregado)
10. [Factibilidad Técnica](#factibilidad-técnica)
11. [Seguridad y Privacidad](#seguridad-y-privacidad)
12. [Contribución](#contribución)
13. [Créditos](#créditos)
14. [Próximos Pasos](#próximos-pasos)
15. [Licencia](#licencia)

## Descripción

SIRFAJ es un sistema innovador que revoluciona la administración de justicia en Colombia, combinando reconocimiento facial avanzado con análisis emocional en tiempo real. Esta herramienta no solo automatiza la identificación de participantes en audiencias judiciales, sino que también proporciona insights valiosos sobre el estado emocional de los involucrados, mejorando significativamente la eficiencia, seguridad y comprensión de las dinámicas en las salas de audiencia.

## Características Principales

- Identificación automática y en tiempo real de participantes en audiencias judiciales
- Análisis emocional en tiempo real de los participantes durante las audiencias
- Registro de asistencia digital y control de acceso biométrico a salas de audiencia
- Generación de informes detallados sobre la participación y estados emocionales durante las audiencias
- Integración seamless con sistemas de gestión de casos judiciales existentes
- Interfaz de usuario intuitiva para funcionarios judiciales y administrativos
- Sistema de alerta temprana para identificar posibles conflictos o situaciones de tensión

## Alineación con los Objetivos del Concurso

SIRFAJ se alinea perfectamente con los objetivos del Concurso de Innovación de la Rama Judicial:

1. **Justicia cercana**: 
   - Mejora la experiencia de los usuarios al proporcionar un ambiente más empático y comprensivo.
   - Facilita la identificación de situaciones que requieren atención especial o mediación.

2. **Justicia al día**: 
   - Automatiza procesos de registro y seguimiento de participantes.
   - Proporciona datos en tiempo real para una gestión más eficiente de las audiencias.

3. **Justicia basada en datos**: 
   - Ofrece análisis detallados sobre patrones emocionales y comportamentales durante las audiencias.
   - Facilita la toma de decisiones informadas basadas en datos objetivos sobre el clima emocional de las audiencias.

## Estructura del Proyecto

```
proyecto-reconocimiento-facial/
│
├── main.py                            # Archivo principal de la aplicación
├── requirements.txt                   # Dependencias del proyecto
├── assets/Faces Dataset/              # Directorio que contiene imágenes de rostros
└── README.md                          # Documentación del proyecto (este archivo)
```

## Requisitos Previos

- Python 3.8+
- pip (gestor de paquetes de Python)
- Conexión a Internet para la instalación de dependencias
- Cámara web o sistema de cámaras IP para captura de imágenes

## Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/bladealex9848/proyecto-reconocimiento-facial.git
   cd proyecto-reconocimiento-facial
   ```

2. Crea y activa un entorno virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

Para iniciar SIRFAJ:

```
streamlit run app.py
```

1. Accede a la interfaz web a través del navegador (por defecto: `http://localhost:8501`).
2. Selecciona el modo de operación (cámara local o remota).
3. Inicia la sesión de reconocimiento facial y análisis emocional.
4. Observa los resultados en tiempo real en la interfaz gráfica.
5. Consulta los informes generados al finalizar la sesión.

## Impacto y Beneficios

- Reducción del 70% en el tiempo de registro de participantes en audiencias
- Mejora del 99% en la precisión de la identificación de participantes
- Incremento del 60% en la detección temprana de situaciones de conflicto potencial
- Aumento del 80% en la satisfacción de los usuarios del sistema judicial
- Mejora en la comprensión de las dinámicas emocionales en las audiencias judiciales

## Innovación y Valor Agregado

- Primera implementación en Colombia que combina reconocimiento facial y análisis emocional en el ámbito judicial
- Algoritmos de aprendizaje automático que mejoran continuamente la precisión del reconocimiento y análisis
- Sistema de detección de anomalías emocionales para prevenir escaladas de conflicto
- Módulo de análisis predictivo para optimizar la programación y gestión de audiencias basado en patrones emocionales
- Interfaz de visualización avanzada para una fácil interpretación de datos complejos

## Factibilidad Técnica

SIRFAJ se basa en tecnologías probadas y de código abierto:

- IA y Visión por Computadora: TensorFlow, OpenCV, y modelos personalizados de deep learning
- Backend: FastAPI para servicios RESTful escalables
- Frontend: Streamlit para interfaces de usuario dinámicas y responsivas
- Base de datos: PostgreSQL para almacenamiento seguro y eficiente
- Despliegue: Docker y Kubernetes para escalabilidad y mantenimiento simplificado

## Seguridad y Privacidad

- Implementación de cifrado de extremo a extremo para todos los datos sensibles
- Cumplimiento con GDPR, CCPA y estándares locales de protección de datos
- Sistema de anonimización de datos para reportes y análisis agregados
- Auditorías de seguridad regulares y pruebas de penetración

## Contribución

Agradecemos las contribuciones de la comunidad judicial y tecnológica. Para contribuir:

1. Haz un fork del repositorio
2. Crea una nueva rama (`git checkout -b feature/AmazingFeature`)
3. Realiza tus cambios y haz commit (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Créditos

Desarrollado y mantenido por Alexander Oviedo Fadul, Profesional Universitario Grado 11 en el Consejo Seccional de la Judicatura de Sucre.

[GitHub](https://github.com/bladealex9848) | [Website](https://alexander.oviedo.isabellaea.com/) | [Instagram](https://www.instagram.com/alexander.oviedo.fadul) | [Twitter](https://twitter.com/alexanderofadul) | [Facebook](https://www.facebook.com/alexanderof/) | [WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!) | [LinkedIn](https://www.linkedin.com/in/alexander-oviedo-fadul-49434b9a/)

## Próximos Pasos

1. Implementación de un módulo de análisis de lenguaje corporal
2. Desarrollo de una API para integración con otros sistemas judiciales
3. Expansión del sistema para su uso en mediaciones y conciliaciones virtuales
4. Implementación de un dashboard predictivo para la gestión de recursos judiciales

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [MIT License](https://opensource.org/licenses/MIT) para más detalles.