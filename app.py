import torch

class DummyPath:
    _path = []

torch.classes.__path__ = DummyPath()

import streamlit as st
import tempfile
from api_handler import get_video_prediction
import os
import plotly.graph_objects as go

# ────── Page Config ──────
st.set_page_config(page_title="Deepfake Video Detector", page_icon="🎥", layout="wide")

# ────── Sticky Header ──────
st.markdown("""
    <style>
        .sticky-header {
            position: sticky;
            top: 0;
            background-color: #0e1117;
            padding: 1rem 0.5rem;
            z-index: 999;
            border-bottom: 1px solid #333;
        }
        .sticky-header h1 {
            font-size: 1.8rem;
            margin: 0;
            color: #FAFAFA;
        }
    </style>
    <div class="sticky-header">
        <h1>🎥 Deepfake Video Detection</h1>
    </div>
""", unsafe_allow_html=True)

# ────── Tabs ──────
tab1, tab2, tab3 = st.tabs(["🔍 Detect Deepfake", "📖 About", "❓ Help"])

with tab1:
    st.markdown("Upload a **video file** (MP4) to detect if it's deepfaked using our AI model.")
    st.set_option('server.maxUploadSize', 50)

    file = st.file_uploader("📤 Upload a video file", type=["mp4"], label_visibility="visible")

    if file:
        file_bytes = file.read()
        file_name = file.name

        MAX_SIZE_MB = 50
        if len(file_bytes) > MAX_SIZE_MB * 1024 * 1024:
            st.error(f"❌ File size exceeds {MAX_SIZE_MB}MB limit. Please upload a smaller video.")
            st.stop()

        videoPath = os.path.join("test_data", file_name)
        os.makedirs("test_data", exist_ok=True)
        with open(videoPath, "wb") as buffer:
            buffer.write(file_bytes)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("🎞️ Video Preview")
            st.video(videoPath)

        with col2:
            st.subheader("🔬 Detection Result")
            st.markdown("*This may take ~10 seconds depending on video length.*")
            with st.spinner("Analyzing video..."):
                result = get_video_prediction(videoPath)

            if result:
                st.success("✅ Detection Complete!")

                label = result.get("label", "Unknown")
                processing_time = result.get("time", "Unknown")
                confidence = float(result.get("probability", 0.0))
                display_conf = confidence if label == "Fake" else 1.0 - confidence
                conf_text = f"{label} Confidence: {int(display_conf * 100)}%"

                # ➤ Confidence Gauge
                st.markdown("### 📈 Confidence Meter")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=display_conf * 100,
                    title={'text': f"{label} Confidence"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "indianred"},
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': display_conf * 100
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                # ➤ Summary Table
                st.markdown("### 📝 Detection Summary")
                st.table({
                    "Label": [label],
                    "Confidence": [f"{int(display_conf * 100)}%"],
                    "Processing Time": [f"{processing_time} seconds"],
                    "Filename": [file_name],
                })
            else:
                st.error("❌ Detection Failed. Please try again.")

with tab2:
    st.markdown("""
    ## 📖 About This Project  
    This app detects deepfake **videos** using advanced AI models trained on real and synthetic datasets.
    
    - 🎞️ Frame sampling & temporal analysis  
    - 🧠 Deep learning with paired fake-real difference modeling  
    - 🎯 Output: label (Real/Fake), confidence score, visual cues  
    """)

with tab3:
    st.markdown("""
    ## ❓ Help & Guide  
    1. Switch to the '🔍 Detect Deepfake' tab  
    2. Upload an **MP4** video  
    3. Wait ~10 seconds while it is processed  
    4. View result + confidence indicator

    ⚠️ Max file size: ~50MB depending on backend limits  
    ✅ Works on both desktop and mobile (responsive layout)
    """)
