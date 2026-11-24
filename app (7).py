import cv2
import numpy as np
import os
import json
import joblib
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

st.set_page_config(
    page_title="IA Fitosanitaria - Guayaba Avanzada",
    page_icon="🍂",
    layout="centered"
)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('modelo_svm.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('recomendaciones.json', 'r', encoding='utf-8') as f:
            recs = json.load(f)
        return model, scaler, recs
    except Exception as e:
        st.error(f"Error cargando recursos: {e}")
        return None, None, None

model, scaler, recomendaciones = load_assets()

def remove_background_grabcut(img_pil, target_size=(128, 128)):
    """
    Segmentación robusta de hojas utilizando una combinación de HSV y GrabCut
    para remover fondos complejos del mundo real de manera efectiva.
    """
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    
    # Crear una máscara inicial basada en verde/marrón/amarillo (tonos de hojas)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 10, 10])
    upper_bound = np.array([180, 255, 240])
    mask_hsv = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Configurar GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Definir un rectángulo que encierra la hoja central (asumiendo margen del 10%)
    h, w = img.shape[:2]
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    
    # Inicializar máscara GrabCut
    mask_gc = np.zeros(img.shape[:2], np.uint8)
    
    # Utilizar la máscara HSV para guiar a GrabCut sobre qué es fondo seguro y qué es probable primer plano
    mask_gc[mask_hsv == 0] = cv2.GC_BGD
    mask_gc[mask_hsv == 255] = cv2.GC_PR_FGD
    
    try:
        # Ejecutar GrabCut para refinar los bordes de la hoja
        cv2.grabCut(img, mask_gc, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        # Crear máscara final de primer plano
        mask_final = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    except:
        mask_final = mask_hsv
        
    # Operaciones morfológicas para suavizar imperfecciones del contorno
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)
    
    # Aplicar máscara sobre fondo blanco puro
    background_white = np.ones_like(img) * 255
    result = np.where(mask_final[:, :, None] == 255, img, background_white)
    
    result_resized = cv2.resize(result, target_size)
    return result_resized

def extract_features_streamlit(img_segmented):
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

st.title("🍃 Diagnóstico Fitosanitario de Precisión")
st.write("Monitoreo fitosanitario inteligente con segmentación GrabCut para la remoción robusta de ruidos de fondo en campo.")

uploaded_file = st.file_uploader("Suba una imagen de la hoja afectada", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img_original = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_original, caption="Fotografía de Campo", use_container_width=True)
        
    with st.spinner("Aislando la estructura de la hoja con GrabCut..."):
        img_seg = remove_background_grabcut(img_original)
        img_seg_rgb = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)
        
    with col2:
        st.image(img_seg_rgb, caption="Hoja Procesada (Fondo Blanco)", use_container_width=True)
        
    if model is not None and scaler is not None:
        feats = extract_features_streamlit(img_seg)
        feats_scaled = scaler.transform([feats])
        prediction = model.predict(feats_scaled)[0]
        
        mapping = {
            "healthy": "Guava_Healthy",
            "anthracnose": "Guava_Anthracnose",
            "red_rust": "Guava_Red_Rust",
            "caterpillar_damage": "Guava_Scab",
            "spiraling_whitefly": "Guava_Scab"
        }
        
        rec_key = mapping.get(prediction.lower(), "Guava_Healthy")
        info = recomendaciones.get(rec_key, recomendaciones["Guava_Healthy"])
        
        st.subheader(f"Diagnóstico Estimado: **{info['nombre_comun']}**")
        
        if info['tipo'] == "Saludable":
            st.success("Hojas sin presencia de patologías de importancia económica.")
        else:
            st.warning(f"Presencia de sintomatología de origen: {info['tipo']}")
            
        st.markdown("### 📋 Protocolo de Mitigación Agronómica")
        st.markdown(f"**1. Manejo Cultural:** {info['manejo_cultural']}")
        st.markdown(f"**2. Alternativa Botánica / Ecológica:** {info['alternativa_botanica']}")
        st.markdown(f"**3. Control Químico Responsable:** {info['control_quimico']}")
    else:
        st.error("Error al cargar el cerebro predictivo de la IA (modelo_svm.pkl).")
''')
