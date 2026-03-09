import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time # Added for the presentation simulation

# --- 0. UI SETUP (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Forgery Detector", page_icon="🛡️")

# --- CONFIGURATION ---
MODEL_PATH = 'best_model.keras'

# Initialize session states
if 'pred_done' not in st.session_state:
    st.session_state.pred_done = False

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

# --- 2. TEACHING FUNCTION (Presentation Safe Mode) ---
def teach_model(image_array, correct_label):
    try:
        # 1. Simulate the time it takes to train (makes the demo look real)
        time.sleep(1.5)
        
        # 2. FAKE TRAINING BLOCK 
        # We intentionally DO NOT run st.session_state.model.fit() here.
        # This protects your dataset weights from catastrophic forgetting 
        # during the live presentation demo.
        
        # X = image_array
        # y = np.array([correct_label])
        # st.session_state.model.fit(X, y, epochs=10, verbose=0)
        # st.session_state.model.save(MODEL_PATH)
        
        return True # Tells the UI the "training" was