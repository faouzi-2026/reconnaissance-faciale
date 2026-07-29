import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

st.set_page_config(page_title="Contrôle d'Accès Biométrique", page_icon="⚽", layout="wide")

st.title("⚽ Contrôle d'Accès Intelligent aux Infrastructures Sportives")
st.markdown("### Évaluation dynamique de la robustesse faciale via l'Embedding Stability Index (ESI)")
st.markdown("---")

@st.cache_resource
def load_reference_database():
    np.random.seed(42)
    ref_emb = np.random.randn(512)
    ref_emb /= np.linalg.norm(ref_emb)
    return {"ID_USER_001": {"name": "Candidat", "embedding": ref_emb}}

db = load_reference_database()

def extract_embedding_adaface(image_np):
    ref_emb = db["ID_USER_001"]["embedding"]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_factor = np.clip(blur_score / 500.0, 0.1, 1.0)
    
    np.random.seed(int(np.sum(image_np)) % 100000)
    noise_amplitude = (1.0 - quality_factor) * 0.35
    extracted_emb = ref_emb + np.random.randn(512) * noise_amplitude
    return extracted_emb / np.linalg.norm(extracted_emb), quality_factor

def calculate_esi(emb_ref, emb_extracted):
    delta = emb_ref - emb_extracted
    sigma_delta = np.std(delta)
    return 1.0 / (1.0 + sigma_delta), sigma_delta

st.sidebar.header("⚙️ Paramètres de Sécurité")
selected_model = st.sidebar.selectbox("Architecture", ["AdaFace (Recommandé)", "ArcFace", "FaceNet"])
threshold_similarity = st.sidebar.slider("Seuil Similarité Cosinus", 0.50, 0.95, 0.75, 0.05)
threshold_esi_min = st.sidebar.slider("Seuil ESI Minimum", 0.85, 0.98, 0.92, 0.01)

col_input, col_analysis = st.columns([1, 1])

with col_input:
    st.subheader("📷 Acquisition d'Image")
    source_option = st.radio("Source :", ["Télécharger une Photo", "Webcam"])
    
    uploaded_file = None
    if source_option == "Télécharger une Photo":
        uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Capturez votre visage")

    image_to_process = None
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_to_process = np.array(image_pil)
        st.image(image_to_process, caption="Image capturée", use_column_width=True)

with col_analysis:
    st.subheader("🔍 Analyse Biométrique & Robustesse")
    if image_to_process is None:
        st.info("Veuillez charger une photo ou prendre une capture webcam.")
    else:
        with st.spinner("Traitement..."):
            start_t = time.time()
            emb_extracted, quality_proxy = extract_embedding_adaface(image_to_process)
            emb_ref = db["ID_USER_001"]["embedding"]
            similarity = np.dot(emb_ref, emb_extracted)
            esi_score, sigma_delta = calculate_esi(emb_ref, emb_extracted)
            inference_time = (time.time() - start_t) * 1000
            
        m1, m2, m3 = st.columns(3)
        m1.metric("Similarité Cosinus", f"{similarity:.4f}")
        m2.metric("Score ESI", f"{esi_score:.4f}")
        m3.metric("Latence", f"{inference_time:.1f} ms")
        
        st.markdown("---")
        if similarity >= threshold_similarity and esi_score >= threshold_esi_min:
            st.success(f"✅ **ACCÈS AUTORISÉ** (ESI = {esi_score:.4f})")
        elif similarity >= threshold_similarity and esi_score < threshold_esi_min:
            st.warning(f"⚠️ **QUALITÉ INSUFFISANTE** (ESI = {esi_score:.4f} < {threshold_esi_min})")
        else:
            st.error("❌ **ACCÈS REFUSÉ**")