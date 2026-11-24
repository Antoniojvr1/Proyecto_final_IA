import streamlit as st
import cv2
import numpy as np
import os
import json
import joblib
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# Configuración de página de Streamlit
st.set_page_config(
    page_title="IA Fitosanitaria - Guayaba",
    page_icon="🍂",
    layout="centered"
)

# --- 1. CARGA DE RECURSOS DEL SISTEMA ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('modelo_svm.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('recomendaciones.json', 'r', encoding='utf-8') as f:
            recs = json.load(f)
        return model, scaler, recs
    except Exception as e:
        st.error(f"Error cargando los modelos o base de conocimientos: {e}")
        return None, None, None

model, scaler, recomendaciones = load_assets()

# --- 2. FUNCIONES DE PROCESAMIENTO DE IMAGEN ---
def remove_background_streamlit(img_pil, target_size=(128, 128)):
    """
    Segmenta la hoja usando espacios de color HSV y operaciones morfológicas,
    y reemplaza el fondo exterior con blanco puro.
    """
    img = np.array(img_pil)
    # Convertir de RGB (PIL) a BGR (OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Rangos para segmentar vegetación
    lower_bound = np.array([0, 10, 10])
    upper_bound = np.array([180, 255, 245])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    background_white = np.ones_like(img) * 255
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        result = np.where(clean_mask[:, :, None] == 255, img, background_white)
    else:
        result = np.where(mask[:, :, None] == 255, img, background_white)
        
    result_resized = cv2.resize(result, target_size)
    return result_resized

def extract_features_streamlit(img_segmented):
    # Características de Color (RGB y HSV)
    mean_r = np.mean(img_segmented[:,:,2])
    std_r = np.std(img_segmented[:,:,2])
    mean_g = np.mean(img_segmented[:,:,1])
    std_g = np.std(img_segmented[:,:,1])
    mean_b = np.mean(img_segmented[:,:,0])
    std_b = np.std(img_segmented[:,:,0])
    
    hsv = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:,:,0])
    std_h = np.std(hsv[:,:,0])
    mean_s = np.mean(hsv[:,:,1])
    std_s = np.std(hsv[:,:,1])
    mean_v = np.mean(hsv[:,:,2])
    std_v = np.std(hsv[:,:,2])
    
    # Características de Textura (GLCM)
    gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    
    return [
        mean_r, std_r, mean_g, std_g, mean_b, std_b,
        mean_h, std_h, mean_s, std_s, mean_v, std_v,
        contrast, dissimilarity, homogeneity, energy, correlation, ASM
    ]

# --- 3. DISEÑO DE LA INTERFAZ DE USUARIO ---
st.title("🍃 Diagnóstico Fitosanitario Inteligente")
st.write("Suba una fotografía de una hoja de guayaba para detectar enfermedades mediante IA con remoción de ruido de fondo automática.")

uploaded_file = st.file_uploader("Seleccione una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img_original = Image.open(uploaded_file)
    
    # Layout de previsualización
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_original, caption="Imagen cargada", use_container_width=True)
        
    with st.spinner("Preprocesando imagen y aislando la hoja..."):
        img_seg = remove_background_streamlit(img_original)
        img_seg_rgb = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)
    
    with col2:
        st.image(img_seg_rgb, caption="Hoja segmentada (Sin Fondo)", use_container_width=True)
        
    if model is not None and scaler is not None:
        # Extracción y clasificación
        feats = extract_features_streamlit(img_seg)
        feats_scaled = scaler.transform([feats])
        
        prediction = model.predict(feats_scaled)[0]
        
        # Mapear predicción al formato de recomendaciones
        # Las clases comunes en el dataset mapean a:
        # 'healthy', 'anthracnose', 'red_rust', 'caterpillar_damage', 'spiraling_whitefly'
        mapping = {
            "healthy": "Guava_Healthy",
            "anthracnose": "Guava_Anthracnose",
            "red_rust": "Guava_Red_Rust",
            "caterpillar_damage": "Guava_Scab",
            "spiraling_whitefly": "Guava_Scab"
        }
        
        rec_key = mapping.get(prediction.lower(), "Guava_Healthy")
        info = recomendaciones.get(rec_key, recomendaciones["Guava_Healthy"])
        
        # Desplegar resultados de diagnóstico
        st.subheader(f"Resultado del Análisis: **{info['nombre_comun']}**")
        
        if info['tipo'] == "Saludable":
            st.success("¡Su cultivo se encuentra en excelentes condiciones!")
        else:
            st.warning(f"Alerta de Fitopatología detectada de tipo: {info['tipo']}")
            
        # Ficha Agronómica estructurada
        st.markdown("### 📋 Ficha de Medidas de Mitigación")
        st.markdown(f"**1. Manejo Cultural:** {info['manejo_cultural']}")
        st.markdown(f"**2. Alternativa Botánica:** {info['alternativa_botanica']}")
        st.markdown(f"**3. Control Químico Responsable:** {info['control_quimico']}")
    else:
        st.error("El modelo o el escalador no se cargaron correctamente.")
