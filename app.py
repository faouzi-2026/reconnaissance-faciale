import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

# Configuration de la page Streamlit
st.set_page_config(page_title="Contrôle d'Accès Biométrique Stade", page_icon="⚽", layout="wide")

st.title("⚽ Contrôle d'Accès Intelligent aux Infrastructures Sportives")
st.markdown("### Évaluation de la robustesse faciale via l'Embedding Stability Index (ESI)")
st.markdown("---")

# ==============================================================================
# 1. INITIALISATION DE LA BASE DE DONNÉES EN MÉMOIRE
# ==============================================================================
if 'database' not in st.session_state:
    st.session_state['database'] = {}

def extract_reference_embedding(image_np):
    """Extrait un embedding de référence propre (normalisé L2) à l'enrôlement."""
    np.random.seed(int(np.sum(image_np)) % 100000)
    emb = np.random.randn(512)
    return emb / np.linalg.norm(emb)

def extract_test_embedding(image_np, ref_emb, model_type="AdaFace"):
    """
    Simule l'extraction d'embedding selon la qualité d'image et l'architecture.
    Conforme au comportement du Chapitre 5 (AdaFace, ArcFace, FaceNet).
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_factor = np.clip(blur_score / 500.0, 0.1, 1.0)
    
    diff_factor = (1.0 - quality_factor)
    
    # Sensibilité du modèle selon la théorie du rapport
    if "AdaFace" in model_type:
        noise_level = diff_factor * 0.25 # Marge adaptative : très stable
    elif "ArcFace" in model_type:
        noise_level = diff_factor * 0.45 # Marge angulaire : moyennement stable
    else:
        noise_level = diff_factor * 0.75 # FaceNet : sensible aux dégradations
        
    np.random.seed(int(np.sum(image_np)) % 100000)
    noisy_emb = ref_emb + np.random.randn(512) * noise_level
    return noisy_emb / np.linalg.norm(noisy_emb)

def calculate_esi(emb_ref, emb_extracted):
    """Calcule l'ESI selon la formule 6.6 : 1 / (1 + sigma_delta)."""
    delta = emb_ref - emb_extracted
    sigma_delta = np.std(delta)
    return 1.0 / (1.0 + sigma_delta)

# ==============================================================================
# 2. BARRE LATÉRALE : PARAMÈTRES & CONFIGURATION
# ==============================================================================
st.sidebar.header("⚙️ Configuration du Système")
selected_model = st.sidebar.selectbox("Modèle Biométrique", ["AdaFace (Marge Adaptative)", "ArcFace (Marge Angulaire)", "FaceNet (Triplet Loss)"])

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Seuis de Sécurité")
threshold_similarity = st.sidebar.slider("Seuil Similarité Cosinus", 0.50, 0.95, 0.75, 0.05)
threshold_esi_min = st.sidebar.slider("Seuil ESI Minimum", 0.85, 0.98, 0.92, 0.01)

# ==============================================================================
# 3. INTERFACE PRINCIPALE EN 2 ONGLETS
# ==============================================================================
tab_enrol, tab_gate = st.tabs(["📋 1. Enrôlement (Base de Données)", "🏟️ 2. Portique d'Accès (Test en Direct)"])

# ------------------------------------------------------------------------------
# ONGLET 1 : ENRÔLEMENT DES SUPPORTERS (IMAGE PROPRE)
# ------------------------------------------------------------------------------
with tab_enrol:
    st.subheader("Ajouter un Supporter à la Base de Données (Image Originale / Propre)")
    col_enrol_input, col_enrol_db = st.columns([1, 1])
    
    with col_enrol_input:
        supporter_name = st.text_input("Nom du Supporter / Identifiant :", placeholder="ex: Mehdi / Supporter_01")
        uploaded_ref = st.file_uploader("Photo d'identité nette (Passeport / Billet) :", type=["jpg", "png", "jpeg"], key="ref_upload")
        
        if st.button("Enregistrer le Supporter"):
            if supporter_name and uploaded_ref:
                img_pil = Image.open(uploaded_ref).convert("RGB")
                img_np = np.array(img_pil)
                ref_emb = extract_reference_embedding(img_np)
                
                # Sauvegarde dans la base
                st.session_state['database'][supporter_name] = {
                    'image': img_np,
                    'embedding': ref_emb
                }
                st.success(f"✅ Supporter '{supporter_name}' inscrit avec succès !")
            else:
                st.warning("Veuillez saisir un nom et charger une photo propre.")

    with col_enrol_db:
        st.subheader("Base de Données Actuelle")
        db_size = len(st.session_state['database'])
        st.info(f"Nombre de supporters inscrits : **{db_size}**")
        
        if db_size > 0:
            cols = st.columns(3)
            for idx, (name, data) in enumerate(st.session_state['database'].items()):
                with cols[idx % 3]:
                    st.image(data['image'], caption=name, use_column_width=True)

# ------------------------------------------------------------------------------
# ONGLET 2 : PORTIQUE DE CONTRÔLE D'ACCÈS (IMAGE DEGRADÉE)
# ------------------------------------------------------------------------------
with tab_gate:
    st.subheader("Contrôle d'Accès au Portique du Stade")
    
    if len(st.session_state['database']) == 0:
        st.error("⚠️ La base de données est vide. Veuillez inscrire au moins un supporter dans l'onglet 'Enrôlement'.")
    else:
        col_capture, col_decision = st.columns([1, 1])
        
        with col_capture:
            st.markdown("#### 📷 Capture au Portique")
            source_option = st.radio("Mode d'acquisition :", ["Télécharger une Photo", "Webcam"], key="gate_src")
            
            gate_file = None
            if source_option == "Télécharger une Photo":
                gate_file = st.file_uploader("Photo capturée (Masque, Bruit, Flou...) :", type=["jpg", "png", "jpeg"], key="gate_upload")
            else:
                gate_file = st.camera_input("Capturez le visage au portique")
                
            gate_image_np = None
            if gate_file is not None:
                img_pil = Image.open(gate_file).convert("RGB")
                gate_image_np = np.array(img_pil)
                st.image(gate_image_np, caption="Visage capturé au portique", use_column_width=True)

        with col_decision:
            st.markdown("#### 🔍 Recherche & Analyse de Stabilité")
            if gate_image_np is not None:
                with st.spinner("Recherche dans la base de données..."):
                    start_time = time.time()
                    
                    best_match_name = None
                    best_similarity = -1.0
                    best_extracted_emb = None
                    
                    # 1. Recherche de l'identité la plus proche dans la DB
                    for name, data in st.session_state['database'].items():
                        ref_emb = data['embedding']
                        # Extraction de l'embedding pour l'image de test
                        test_emb = extract_test_embedding(gate_image_np, ref_emb, selected_model)
                        sim = np.dot(ref_emb, test_emb)
                        
                        if sim > best_similarity:
                            best_similarity = sim
                            best_match_name = name
                            best_extracted_emb = test_emb
                            
                    inference_time = (time.time() - start_time) * 1000
                    
                    # 2. Calcul du score ESI pour l'identité trouvée
                    ref_emb_matched = st.session_state['database'][best_match_name]['embedding']
                    esi_score = calculate_esi(ref_emb_matched, best_extracted_emb)
                    
                    # Affichage de l'identité associée
                    st.markdown(f"**Identité la plus proche :** `{best_match_name}`")
                    
                    # Affichage des Métriques
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Similarité Cosinus", f"{best_similarity:.4f}")
                    m2.metric("Score ESI", f"{esi_score:.4f}")
                    m3.metric("Latence", f"{inference_time:.1f} ms")
                    
                    st.markdown("---")
                    
                    # 3. Prise de Décision à 3 Niveaux
                    if best_similarity >= threshold_similarity and esi_score >= threshold_esi_min:
                        st.success(f"✅ **ACCÈS AUTORISÉ**\n\nBienvenue **{best_match_name}** ! La représentation est stable (ESI = {esi_score:.4f}).")
                    elif best_similarity >= threshold_similarity and esi_score < threshold_esi_min:
                        st.warning(f"⚠️ **QUALITÉ INSUFFISANTE (ESI = {esi_score:.4f} < {threshold_esi_min})**\n\nIdentité présumée : **{best_match_name}**.\n\n*Suggestion : L'embedding est trop instable (masque ou flou important). Veuillez effectuer une seconde capture.*")
                    else:
                        st.error("❌ **ACCÈS REFUSÉ**\n\nIdentité non reconnue dans la base autorisée.")
            else:
                st.info("En attente d'une capture au portique...")
