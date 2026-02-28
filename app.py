import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="",
    layout="wide"
)

# ========== KUSTOMISASI TAMPILAN ==========
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Gradient background untuk header */
    .gradient-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .gradient-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .gradient-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Card styling untuk hasil prediksi */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .result-card.fraud {
        border-left-color: #dc3545;
        background: #fff5f5;
    }
    
    .result-card.normal {
        border-left-color: #28a745;
        background: #f0fff0;
    }
    
    .result-card h2 {
        margin: 0 0 1rem 0;
        font-size: 1.8rem;
    }
    
    .result-card .probability {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .result-card progress {
        width: 100%;
        height: 25px;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    progress.fraud {
        accent-color: #dc3545;
    }
    
    progress.normal {
        accent-color: #28a745;
    }
    
    /* Metric cards styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        flex: 1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-card .label {
        font-size: 0.9rem;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2d3748;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .sidebar-info h4 {
        margin: 0 0 0.8rem 0;
        color: #4a5568;
    }
    
    .sidebar-info p {
        margin: 0.3rem 0;
        color: #2d3748;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: bold;
        color: #4a5568;
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Ganti judul dengan header kustom
st.markdown("""
<div class="gradient-header">
    <h1>FRAUD DETECTION SYSTEM</h1>
    <p>Real-time Credit Card Fraud Detection menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Load model dan scaler
@st.cache_resource
def load_model():
    model = joblib.load('best_fraud_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-info" style="background: #28a745; color: white;">
            <p style="margin:0; color: white;">Model dan Scaler berhasil diload</p>
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-info" style="background: #dc3545; color: white;">
            <p style="margin:0; color: white;">Error loading model: {e}</p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# Sidebar untuk informasi
with st.sidebar:
    st.markdown("### Informasi Model")
    st.markdown("""
    <div class="sidebar-info">
        <h4>Model Random Forest</h4>
        <p>Test Recall: 79.7%</p>
        <p>Test F1-Score: 0.747</p>
        <p>Test ROC-AUC: 0.979</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Statistik Dataset")
    st.markdown("""
    <div class="sidebar-info">
        <p>Total transaksi: 284.807</p>
        <p>Fraud: 492 (0.17%)</p>
        <p>Normal: 284.315 (99.83%)</p>
    </div>
    """, unsafe_allow_html=True)

# Main content - dibagi 2 kolom
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Data Transaksi")
    
    # Input waktu
    with st.expander("Waktu Transaksi", expanded=True):
        time_hour = st.number_input("Jam (0-23)", min_value=0, max_value=23, value=14)
        time_minute = st.number_input("Menit (0-59)", min_value=0, max_value=59, value=30)
        time_second = st.number_input("Detik (0-59)", min_value=0, max_value=59, value=0)
        
        # Konversi ke detik
        time_seconds = time_hour * 3600 + time_minute * 60 + time_second
    
    # Input amount
    with st.expander("Amount (Jumlah Uang)", expanded=True):
        amount = st.number_input("Amount (Rp)", min_value=0.0, max_value=100000000.0, value=1000000.0, step=10000.0)
        st.caption("Contoh: Rp 1.000.000 = 1000000")
    
    # Input fitur PCA
    with st.expander("Fitur PCA (V1 - V28)", expanded=False):
        st.caption("Masukkan nilai fitur PCA (biasanya antara -10 sampai 10)")
        
        cols = st.columns(2)
        v_features = {}
        
        for i in range(1, 29):
            with cols[0] if i % 2 == 1 else cols[1]:
                v_features[f'V{i}'] = st.number_input(
                    f'V{i}',
                    value=0.0,
                    format="%.4f",
                    key=f'v{i}'
                )
    
    # Tombol prediksi
    predict_button = st.button("CEK FRAUD", type="primary", use_container_width=True)

with col2:
    st.header("Hasil Prediksi")
    
    if predict_button:
        # ============================================
        # MEMBANGUN FITUR ARRAY
        # ============================================
        # Urutan fitur: Time, V1, V2, ..., V28, Amount
        features = []
        
        # 1. Time
        features.append(time_seconds)
        
        # 2. V1 sampai V28
        for i in range(1, 29):
            features.append(v_features[f'V{i}'])
        
        # 3. Amount
        features.append(amount)
        
        # Konversi ke numpy array (1 baris, 30 kolom)
        features_array = np.array([features])
        
        # ============================================
        # NORMALISASI (HANYA TIME DAN AMOUNT)
        # ============================================
        # Ambil Time dan Amount (index 0 dan index 29)
        time_amount = features_array[:, [0, -1]]  # Index 0 (Time) dan index terakhir (Amount)
        
        # Normalisasi dengan scaler
        time_amount_scaled = scaler.transform(time_amount)
        
        # Gabungkan kembali dengan V1-V28
        features_scaled = features_array.copy()
        features_scaled[:, 0] = time_amount_scaled[:, 0]   # Ganti Time dengan hasil normalisasi
        features_scaled[:, -1] = time_amount_scaled[:, 1]  # Ganti Amount dengan hasil normalisasi
        
        # ============================================
        # PREDIKSI
        # ============================================
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # ============================================
        # TAMPILKAN HASIL
        # ============================================
        st.markdown("### HASIL ANALISIS:")
        
        # Metric cards
        st.markdown("""
        <div class="metric-container">
        """, unsafe_allow_html=True)
        
        col2a, col2b, col2c = st.columns(3)
        with col2a:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Amount</div>
                <div class="value">Rp {amount:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Waktu</div>
                <div class="value">{time_hour:02d}:{time_minute:02d}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2c:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Probabilitas Fraud</div>
                <div class="value">{probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Hasil prediksi dengan card
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card fraud">
                <h2>FRAUD TERDETEKSI</h2>
                <div class="probability">Probabilitas: {probability:.1%}</div>
                <progress class="fraud" value="{probability}" max="1.0"></progress>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Rekomendasi Tindakan"):
                st.markdown("""
                - Blokir sementara transaksi
                - Hubungi pemilik kartu untuk verifikasi
                - Catat IP address dan lokasi
                - Laporkan ke tim fraud
                """)
        else:
            st.markdown(f"""
            <div class="result-card normal">
                <h2>TRANSAKSI NORMAL</h2>
                <div class="probability">Probabilitas Fraud: {probability:.1%}</div>
                <progress class="normal" value="{probability}" max="1.0"></progress>
            </div>
            """, unsafe_allow_html=True)
        
        # Tampilkan detail fitur
        with st.expander("Detail Fitur Input"):
            # Buat list nama fitur
            feature_names = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
            
            df_detail = pd.DataFrame({
                'Fitur': feature_names,
                'Nilai Input': features,
                'Nilai Scaled': features_scaled[0]
            })
            st.dataframe(df_detail, use_container_width=True)
    else:
        st.info("Masukkan data transaksi di kolom kiri, lalu klik 'CEK FRAUD'")
        st.markdown("### Contoh Hasil Prediksi")
        st.image("https://via.placeholder.com/400x200?text=Hasil+Prediksi+Akan+Muncul+di+Sini")

# Footer
st.markdown("""
<div class="footer">
    <p>Fraud Detection System | Model Random Forest | Recall: 79.7%</p>
    <p>Dibuat oleh Septiyano | 2026</p>
</div>
""", unsafe_allow_html=True)