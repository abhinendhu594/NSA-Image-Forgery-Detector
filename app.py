import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = 'best_model.keras'

# --- 1. LOAD MODEL (Stable Mode) ---
@st.cache_resource
def load_model():
    try:
        # Load the base structure
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        
        # FREEZE the brain so it doesn't get confused
        base_model.trainable = False 
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Load your trained weights
        model.load_weights(MODEL_PATH)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if 'model' not in st.session_state:
    st.session_state.model = load_model()

# --- 2. TEACHING FUNCTION (The Stabilizer) ---
def teach_model(image_array, correct_label):
    try:
        # Only let it change the final decision layer
        for layer in st.session_state.model.layers[:-1]:
            layer.trainable = False
        st.session_state.model.layers[-1].trainable = True
        
        # Use a FRESH optimizer every time to prevent crashing
        new_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        st.session_state.model.compile(
            optimizer=new_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train gently (10 epochs)
        X = image_array
        y = np.array([correct_label])
        st.session_state.model.fit(X, y, epochs=10, verbose=0)
        
        # Save the improved brain
        st.session_state.model.save(MODEL_PATH)
        return True
    except Exception as e:
        st.error(f"Training Error: {e}")
        return False

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="AI Forgery Detector", page_icon="🛡️")

st.title("🛡️ AI Forgery Detection System")
st.markdown("**System Status:** `Online` | **Mode:** `Forensic Analysis`")

uploaded_file = st.file_uploader("Upload Image for Analysis", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    # Fix transparency issues
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Evidence', use_container_width=True)
    
    # Preprocessing
    img_tensor = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_tensor).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction Button
    if st.button("🔍 Run Forensic Analysis"):
        with st.spinner('Scanning pixel artifacts...'):
            prediction = st.session_state.model.predict(img_array)[0][0]
            
            confidence = prediction if prediction > 0.5 else 1 - prediction
            label = "REAL" if prediction > 0.5 else "FAKE / GENERATED"
            color = "#00c853" if label == "REAL" else "#ff1744"

            st.markdown(f"""
            <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border: 2px solid {color}; text-align: center;">
                <h2 style="color: {color}; margin: 0;">{label}</h2>
                <p style="margin: 0; font-size: 18px;">Confidence: <strong>{confidence:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.last_img = img_array
            st.session_state.pred_done = True

# --- 4. FEEDBACK LOOP (Active Learning) ---
if 'pred_done' in st.session_state and st.session_state.pred_done:
    st.divider()
    st.write("### 🧠 Correction Mode (Active Learning)")
    st.info("If the result is wrong, teach the model the truth below.")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Actually FAKE"):
            if teach_model(st.session_state.last_img, 0):
                st.success("Analysis Updated: Model has learned this pattern is FAKE.")
                st.session_state.pred_done = False
            
    with c2:
        if st.button("Actually REAL"):
            if teach_model(st.session_state.last_img, 1):
                st.success("Analysis Updated: Model has learned this pattern is REAL.")
                st.session_state.pred_done = False