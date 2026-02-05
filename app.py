"""
===============================================================================
                    CYBERGUARD AI: ENTERPRISE EDITION
               Sistem Deteksi Intrusi Jaringan (Network IDS)
===============================================================================
Versi: 3.0.0 - Hybrid Simulation Mode
Deskripsi: Aplikasi IDS dengan Machine Learning dan MikroTik Integration
===============================================================================
"""

import warnings
import io
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import time

from languages import get_text, TRANSLATIONS
from icons import get_icon, icon_with_text, ICONS
from mikrotik_api import (
    MikroTikConnection, 
    parse_connection_for_ai, 
    get_connection_stats
)

warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(
    page_title="CyberGuard AI: Enterprise Edition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Glassmorphism
CUSTOM_CSS = """
<style>
    .stApp { background: linear-gradient(135deg, #0E1117 0%, #1a1f2e 50%, #0E1117 100%); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 15px rgba(0, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 25px rgba(0, 255, 255, 0.2);
    }
    div[data-testid="metric-container"] label { color: #00ffff !important; font-weight: 600; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { 
        color: #ffffff !important; font-size: 2rem !important; font-weight: 700; 
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0d12 0%, #151921 100%);
        border-right: 1px solid rgba(0, 255, 255, 0.2);
    }
    
    .nav-item {
        display: flex; align-items: center; gap: 12px;
        padding: 14px 18px; margin: 6px 0;
        background: rgba(0, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 255, 0.15);
        border-radius: 10px;
        color: #ffffff; cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-item:hover {
        background: rgba(0, 255, 255, 0.15);
        border-color: rgba(0, 255, 255, 0.4);
        transform: translateX(5px);
    }
    .nav-item.active {
        background: rgba(0, 255, 255, 0.2);
        border-color: #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    
    h1, h2, h3 { color: #ffffff !important; }
    h1 {
        font-size: 2.5rem !important; font-weight: 800 !important;
        background: linear-gradient(90deg, #00ffff, #00ff88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00ffff 0%, #00cc99 100%);
        color: #0E1117 !important; font-weight: 700; border: none;
        border-radius: 12px; padding: 12px 28px;
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-3px); }
    
    .cyber-header {
        text-align: center; padding: 20px; margin-bottom: 30px;
        background: rgba(0, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 20px;
    }
    
    .threat-normal {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 255, 136, 0.05));
        border: 2px solid #00ff88; border-radius: 16px; padding: 30px; text-align: center;
    }
    .threat-attack {
        background: linear-gradient(135deg, rgba(255, 76, 76, 0.2), rgba(255, 76, 76, 0.05));
        border: 2px solid #ff4c4c; border-radius: 16px; padding: 30px; text-align: center;
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 30px rgba(255, 76, 76, 0.3); }
        50% { box-shadow: 0 0 50px rgba(255, 76, 76, 0.5); }
    }
    
    .simulation-badge {
        background: linear-gradient(135deg, #ff9500, #ff6b00);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 10px;
    }
    
    .live-badge {
        background: linear-gradient(135deg, #00ff88, #00cc66);
        color: #0E1117;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 10px;
        animation: pulse-green 2s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
        50% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.8); }
    }
    
    .connection-table {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 10px;
    }
    
    hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.5), transparent); margin: 30px 0; }
    
    .lang-selector { margin-top: 10px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# FUNGSI DATA & MODEL
# =============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIR, 'dataset', 'KDDTrain+.txt')

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """Memuat dan memproses dataset NSL-KDD dari file lokal."""
    try:
        column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'class', 'difficulty_level'
        ]
        
        df = pd.read_csv(DATASET_PATH, names=column_names, header=None)
        
        selected_columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'class']
        df_selected = df[selected_columns].copy()
        
        categorical_columns = ['protocol_type', 'service', 'flag']
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df_selected[col] = le.fit_transform(df_selected[col].astype(str))
            label_encoders[col] = le
        
        df_selected['class'] = df_selected['class'].apply(lambda x: 0 if x == 'normal' else 1)
        feature_names = [col for col in selected_columns if col != 'class']
        X = df_selected[feature_names]
        y = df_selected['class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test, label_encoders, feature_names
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None, None, None, None

@st.cache_resource(show_spinner=False)
def train_model(_X_train, _y_train):
    """Melatih model Random Forest."""
    model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, 
                                   min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(_X_train, _y_train)
    return model

@st.cache_data(show_spinner=False)
def evaluate_model(_model, _X_test, _y_test):
    """Mengevaluasi model."""
    y_pred = _model.predict(_X_test)
    y_proba = _model.predict_proba(_X_test)
    return {
        'accuracy': accuracy_score(_y_test, y_pred),
        'precision': precision_score(_y_test, y_pred, average='weighted'),
        'recall': recall_score(_y_test, y_pred, average='weighted'),
        'f1': f1_score(_y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(_y_test, y_pred),
        'y_pred': y_pred, 'y_proba': y_proba
    }

def generate_live_traffic_data():
    """Generate data simulasi traffic."""
    now = datetime.now()
    timestamps = [now - timedelta(hours=i) for i in range(24, 0, -1)]
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    base = np.array([100, 80, 60, 40, 30, 25, 30, 50, 120, 180, 220, 250, 240, 200, 180, 160, 170, 190, 220, 200, 180, 160, 140, 120])
    traffic = np.maximum(base + np.random.normal(0, 20, 24), 10)
    return pd.DataFrame({'timestamp': timestamps, 'packets': traffic.astype(int)})

def predict_connection(model, label_encoders, connection_data: dict) -> tuple:
    """
    Predict if a connection is normal or attack.
    
    Returns: (prediction, confidence, probabilities)
    """
    try:
        # Encode categorical features
        protocol_enc = label_encoders['protocol_type'].transform([connection_data['protocol_type']])[0] \
            if connection_data['protocol_type'] in label_encoders['protocol_type'].classes_ else 0
        service_enc = label_encoders['service'].transform([connection_data['service']])[0] \
            if connection_data['service'] in label_encoders['service'].classes_ else 0
        flag_enc = label_encoders['flag'].transform([connection_data['flag']])[0] \
            if connection_data['flag'] in label_encoders['flag'].classes_ else 0
        
        # Prepare feature array
        features = np.array([[
            connection_data.get('duration', 0),
            protocol_enc,
            service_enc,
            flag_enc,
            connection_data.get('src_bytes', 0),
            connection_data.get('dst_bytes', 0),
            connection_data.get('count', 1),
            connection_data.get('srv_count', 1)
        ]])
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        return prediction, confidence, probabilities
    except Exception as e:
        return 0, 0.5, [0.5, 0.5]

# =============================================================================
# HALAMAN SOC MONITORING (BARU)
# =============================================================================

def render_soc_monitoring_page(model, label_encoders, mikrotik_conn, lang):
    """Render SOC Real-Time Monitoring Dashboard."""
    t = lambda k: get_text(k, lang)
    
    simulation_mode = st.session_state.get('simulation_mode', True)
    mode_badge = '<span class="simulation-badge">SIMULATION</span>' if simulation_mode else '<span class="live-badge">LIVE</span>'
    
    st.markdown(f"""
    <div class="cyber-header">
        <h1>SOC Monitoring {mode_badge}</h1>
        <p style="color: #00ffff; font-size: 1.1rem;">Real-Time Network Traffic Analysis • AI-Powered Threat Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh toggle
    col_refresh, col_interval = st.columns([2, 1])
    with col_refresh:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True, key="auto_refresh")
    with col_interval:
        refresh_interval = st.selectbox("Interval", [5, 10, 30, 60], index=0, format_func=lambda x: f"{x} detik")
    
    # Get connections from MikroTik (simulation or real)
    connections = mikrotik_conn.get_firewall_connections()
    stats = get_connection_stats(connections)
    
    # Parse and predict all connections
    analyzed_connections = []
    attack_count = 0
    normal_count = 0
    
    for conn in connections:
        parsed = parse_connection_for_ai(conn)
        prediction, confidence, probs = predict_connection(model, label_encoders, parsed)
        
        parsed['prediction'] = 'Attack' if prediction == 1 else 'Normal'
        parsed['confidence'] = confidence
        parsed['attack_prob'] = probs[1]
        
        if prediction == 1:
            attack_count += 1
        else:
            normal_count += 1
        
        analyzed_connections.append(parsed)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Connections", f"{stats['total']:,}")
    with c2:
        st.metric("Normal Traffic", f"{normal_count:,}", f"{(normal_count/max(1,stats['total'])*100):.1f}%")
    with c3:
        st.metric("🚨 Threats Detected", f"{attack_count:,}", f"{(attack_count/max(1,stats['total'])*100):.1f}%", delta_color="inverse")
    with c4:
        st.metric("Unique Sources", f"{stats['unique_sources']:,}")
    with c5:
        st.metric("Data Transfer", f"{(stats['total_bytes_in'] + stats['total_bytes_out']) / 1024 / 1024:.2f} MB")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Charts
    chart1, chart2 = st.columns(2)
    
    with chart1:
        st.subheader("📊 Protocol Distribution")
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        protocols = ['TCP', 'UDP', 'ICMP']
        values = [stats['tcp'], stats['udp'], stats['icmp']]
        colors = ['#00ffff', '#00ff88', '#ff9500']
        bars = ax.bar(protocols, values, color=colors, edgecolor='white', linewidth=2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, f'{val}',
                   ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white', labelsize=11)
        ax.spines['bottom'].set_color('#333'); ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with chart2:
        st.subheader("🎯 Threat Distribution")
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        colors = ['#00ff88', '#ff4c4c']
        wedges, texts, autotexts = ax.pie(
            [normal_count, attack_count], 
            labels=['Normal', 'Attack'],
            colors=colors, 
            explode=(0.02, 0.08), 
            autopct='%1.1f%%', 
            startangle=90, 
            textprops={'color': 'white', 'fontsize': 12}
        )
        for a in autotexts: 
            a.set_fontweight('bold')
            a.set_fontsize(14)
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Connection Table with Threat Highlighting
    st.subheader("🔍 Live Connection Analysis")
    
    # Filter options
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        show_filter = st.selectbox("Filter", ["All", "Attacks Only", "Normal Only"])
    
    # Prepare dataframe
    if analyzed_connections:
        df_connections = pd.DataFrame(analyzed_connections)
        
        # Apply filter
        if show_filter == "Attacks Only":
            df_connections = df_connections[df_connections['prediction'] == 'Attack']
        elif show_filter == "Normal Only":
            df_connections = df_connections[df_connections['prediction'] == 'Normal']
        
        # Select columns for display
        display_columns = ['src_address', 'dst_address', 'protocol_type', 'src_bytes', 'count', 'prediction', 'confidence']
        df_display = df_connections[display_columns].copy()
        df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.2%}")
        status_col = t('status')
        df_display.columns = [t('source'), t('destination'), t('protocol'), t('bytes'), t('count'), status_col, t('confidence')]
        
        # Style the dataframe
        def highlight_attacks(row):
            if row[status_col] == 'Attack':
                return ['background-color: rgba(255, 76, 76, 0.3)'] * len(row)
            return [''] * len(row)
        
        styled_df = df_display.head(50).style.apply(highlight_attacks, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Alert section for attacks
        attack_df = df_connections[df_connections['prediction'] == 'Attack']
        if len(attack_df) > 0:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("🚨 Active Threat Alerts")
            
            # Group by source IP
            for idx, row in attack_df.head(10).iterrows():
                src_ip = row['src_address'].split(':')[0]
                dst_ip = row['dst_address'].split(':')[0]
                conn_type = row.get('connection_type', 'unknown')
                
                alert_color = "#ff4c4c" if row['confidence'] > 0.8 else "#ff9500"
                st.markdown(f"""
                <div style="background: rgba(255, 76, 76, 0.1); border-left: 4px solid {alert_color}; 
                            padding: 15px; margin: 10px 0; border-radius: 8px;">
                    <strong style="color: {alert_color};">⚠️ Potential {conn_type.upper()} Attack</strong><br>
                    <span style="color: #aaa;">Source: {src_ip} → Destination: {dst_ip}</span><br>
                    <span style="color: #fff;">Bytes: {row['src_bytes']:,} | Count: {row['count']} | Confidence: {row['confidence']:.2%}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No connections to display")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(0.1)  # Small delay to prevent too fast refresh
        st.rerun()

# =============================================================================
# HALAMAN DASHBOARD
# =============================================================================

def render_dashboard_page(model, metrics, X_test, y_test, lang):
    t = lambda k: get_text(k, lang)
    
    st.markdown(f"""
    <div class="cyber-header">
        <h1>{t('dashboard_title')}</h1>
        <p style="color: #00ffff; font-size: 1.1rem;">{t('dashboard_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    total_samples = len(y_test)
    attack_count = int(y_test.sum())
    normal_count = total_samples - attack_count
    threat_pct = (attack_count / total_samples) * 100
    
    if threat_pct < 20: threat_level = t('threat_low')
    elif threat_pct < 50: threat_level = t('threat_moderate')
    else: threat_level = t('threat_high')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(t('status_label'), t('status_online'), t('active_monitoring'))
    with col2: st.metric(t('total_packets'), f"{total_samples:,}", f"+{int(total_samples*0.05):,} {t('new_packets')}")
    with col3: st.metric(t('threat_level'), threat_level, f"{threat_pct:.1f}% {t('threats')}")
    with col4: st.metric(t('model_accuracy'), f"{metrics['accuracy']:.2%}", t('trained_ready'))
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader(t('live_traffic'))
        traffic_data = generate_live_traffic_data()
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(traffic_data['timestamp'], traffic_data['packets'], color='#00ffff', linewidth=2, marker='o', markersize=4)
        ax.fill_between(traffic_data['timestamp'], traffic_data['packets'], alpha=0.3, color='#00ffff')
        ax.tick_params(colors='#888888', labelsize=8)
        ax.spines['bottom'].set_color('#333333'); ax.spines['left'].set_color('#333333')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2, color='#00ffff')
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig); plt.close()
    
    with chart2:
        st.subheader(t('traffic_distribution'))
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
        colors = ['#00ff88', '#ff4c4c']
        wedges, texts, autotexts = ax.pie([normal_count, attack_count], labels=[t('normal_traffic'), t('attack_traffic')],
                                          colors=colors, explode=(0.05, 0.05), autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
        for a in autotexts: a.set_fontweight('bold'); a.set_fontsize(14)
        ax.axis('equal')
        ax.legend(wedges, [f"{t('normal_traffic')}: {normal_count:,}", f"{t('attack_traffic')}: {attack_count:,}"],
                  loc='lower right', facecolor='#0E1117', edgecolor='#333333', labelcolor='white')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# =============================================================================
# HALAMAN SIMULASI LANGSUNG
# =============================================================================

def render_simulation_page(model, label_encoders, feature_names, lang):
    t = lambda k: get_text(k, lang)
    
    st.markdown(f"""
    <div class="cyber-header">
        <h1>{t('simulation_title')}</h1>
        <p style="color: #00ffff; font-size: 1.1rem;">{t('simulation_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"> {t('simulation_instructions')}")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.form(key='packet_form'):
        st.subheader(t('packet_details'))
        col1, col2 = st.columns(2)
        with col1:
            protocol = st.selectbox(t('protocol_type'), ['tcp', 'udp', 'icmp'], help=t('protocol_help'))
            service = st.selectbox(t('network_service'), ['http', 'ftp', 'smtp', 'ssh', 'dns', 'telnet', 'pop3', 'private', 'other'], help=t('service_help'))
            flag = st.selectbox(t('connection_flag'), ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH'], help=t('flag_help'))
            duration = st.number_input(t('duration'), min_value=0, max_value=100000, value=0, help=t('duration_help'))
        with col2:
            src_bytes = st.number_input(t('source_bytes'), min_value=0, max_value=10000000, value=0, help=t('src_bytes_help'))
            dst_bytes = st.number_input(t('dest_bytes'), min_value=0, max_value=10000000, value=0, help=t('dst_bytes_help'))
            count = st.number_input(t('connection_count'), min_value=0, max_value=1000, value=1, help=t('count_help'))
            srv_count = st.number_input(t('service_count'), min_value=0, max_value=1000, value=1, help=t('srv_count_help'))
        
        submitted = st.form_submit_button(t('scan_packet'), use_container_width=True)
    
    if submitted:
        with st.spinner(t('analyzing')):
            try:
                protocol_enc = label_encoders['protocol_type'].transform([protocol])[0]
                service_enc = label_encoders['service'].transform([service])[0] if service in label_encoders['service'].classes_ else 0
                flag_enc = label_encoders['flag'].transform([flag])[0] if flag in label_encoders['flag'].classes_ else 0
                
                packet = np.array([[duration, protocol_enc, service_enc, flag_enc, src_bytes, dst_bytes, count, srv_count]])
                pred = model.predict(packet)[0]
                proba = model.predict_proba(packet)[0]
                conf = proba[pred]
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader(t('analysis_result'))
                
                if pred == 1:
                    st.markdown(f"""
                    <div class="threat-attack">
                        <h2 style="color: #ff4c4c;">{get_icon('alert')} {t('threat_detected')}</h2>
                        <p style="color: #fff;">{t('potential_intrusion')}</p>
                        <p style="color: #ff4c4c; font-size: 1.5rem; font-weight: bold;">{t('confidence')}: {conf:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="threat-normal">
                        <h2 style="color: #00ff88;">{get_icon('check')} {t('packet_cleared')}</h2>
                        <p style="color: #fff;">{t('normal_detected')}</p>
                        <p style="color: #00ff88; font-size: 1.5rem; font-weight: bold;">{t('confidence')}: {conf:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1: st.metric(t('normal_prob'), f"{proba[0]:.2%}")
                with c2: st.metric(t('attack_prob'), f"{proba[1]:.2%}")
            except Exception as e:
                st.error(f"{t('analysis_error')}: {str(e)}")

# =============================================================================
# HALAMAN ANALISIS BATCH
# =============================================================================

def render_batch_page(model, label_encoders, feature_names, lang):
    t = lambda k: get_text(k, lang)
    
    st.markdown(f"""
    <div class="cyber-header">
        <h1>{t('batch_title')}</h1>
        <p style="color: #00ffff; font-size: 1.1rem;">{t('batch_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"> {t('batch_instructions')}")
    
    with st.expander(t('expected_columns')):
        st.markdown(f"""
        | {t('column')} | {t('description')} | {t('type')} |
        |--------|-------------|------|
        | duration | Durasi koneksi | Integer |
        | protocol_type | Protokol (tcp/udp/icmp) | String |
        | service | Layanan jaringan | String |
        | flag | Flag koneksi | String |
        | src_bytes | Byte sumber | Integer |
        | dst_bytes | Byte tujuan | Integer |
        | count | Jumlah koneksi | Integer |
        | srv_count | Jumlah layanan | Integer |
        """)
    
    uploaded = st.file_uploader(t('upload_csv'), type=['csv'])
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"{t('file_success')} {len(df)} {t('records')}.")
            st.subheader(t('data_preview'))
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader(t('column_mapping'))
            col_mapping = {}
            cols = st.columns(4)
            for i, feat in enumerate(feature_names):
                with cols[i % 4]:
                    default = 0
                    for j, c in enumerate(df.columns):
                        if feat.lower() in c.lower(): default = j; break
                    col_mapping[feat] = st.selectbox(feat, df.columns.tolist(), index=default, key=f"m_{feat}")
            
            if st.button(t('analyze_logs'), use_container_width=True):
                with st.spinner(t('processing')):
                    try:
                        analysis_df = pd.DataFrame()
                        for feat in feature_names:
                            analysis_df[feat] = df[col_mapping[feat]].copy()
                        
                        for col in ['protocol_type', 'service', 'flag']:
                            if col in analysis_df.columns:
                                analysis_df[col] = analysis_df[col].apply(
                                    lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0])
                                analysis_df[col] = label_encoders[col].transform(analysis_df[col].astype(str))
                        
                        preds = model.predict(analysis_df)
                        probs = model.predict_proba(analysis_df)
                        
                        result_df = df.copy()
                        result_df['Prediction'] = ['Attack' if p == 1 else 'Normal' for p in preds]
                        result_df['Confidence'] = [max(prob) for prob in probs]
                        
                        attack_cnt = sum(preds)
                        normal_cnt = len(preds) - attack_cnt
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.subheader(t('analysis_results'))
                        
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric(t('total_records'), len(preds))
                        with c2: st.metric(t('normal_traffic'), normal_cnt, f"{(normal_cnt/len(preds))*100:.1f}%")
                        with c3: st.metric(t('threats_detected'), attack_cnt, f"{(attack_cnt/len(preds))*100:.1f}%")
                        
                        st.subheader(t('detection_summary'))
                        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
                        ax.set_facecolor('#0E1117')
                        bars = ax.bar([t('normal_traffic'), t('attack_traffic')], [normal_cnt, attack_cnt], 
                                     color=['#00ff88', '#ff4c4c'], edgecolor='white', linewidth=2)
                        for bar, val in zip(bars, [normal_cnt, attack_cnt]):
                            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:,}',
                                   ha='center', va='bottom', color='white', fontsize=14, fontweight='bold')
                        ax.tick_params(colors='white', labelsize=11)
                        ax.spines['bottom'].set_color('#333'); ax.spines['left'].set_color('#333')
                        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                        plt.tight_layout(); st.pyplot(fig); plt.close()
                        
                        st.subheader(t('detailed_results'))
                        st.dataframe(result_df, use_container_width=True)
                        
                        st.subheader(t('download_results'))
                        csv = result_df.to_csv(index=False)
                        st.download_button(t('download_csv'), csv, f"cyberguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                          "text/csv", use_container_width=True)
                    except Exception as e:
                        st.error(f"{t('mapping_error')}: {str(e)}")
        except Exception as e:
            st.error(f"{t('file_error')}: {str(e)}")

# =============================================================================
# HALAMAN PERFORMA AI
# =============================================================================

def render_performance_page(model, metrics, feature_names, lang):
    t = lambda k: get_text(k, lang)
    
    st.markdown(f"""
    <div class="cyber-header">
        <h1>{t('performance_title')}</h1>
        <p style="color: #00ffff; font-size: 1.1rem;">{t('performance_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"> {t('performance_desc')}")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.subheader(t('performance_overview'))
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric(t('accuracy'), f"{metrics['accuracy']:.4f}")
    with c2: st.metric(t('precision'), f"{metrics['precision']:.4f}")
    with c3: st.metric(t('recall'), f"{metrics['recall']:.4f}")
    with c4: st.metric(t('f1_score'), f"{metrics['f1']:.4f}")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t('confusion_matrix'))
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[t('normal_traffic'), t('attack_traffic')],
                   yticklabels=[t('normal_traffic'), t('attack_traffic')],
                   ax=ax, annot_kws={'size': 18, 'weight': 'bold'}, linewidths=2, linecolor='#0E1117')
        ax.set_xlabel('Predicted', color='white', fontsize=12)
        ax.set_ylabel('Actual', color='white', fontsize=12)
        ax.tick_params(colors='white', labelsize=11)
        plt.setp(ax.get_xticklabels(), color='white')
        plt.setp(ax.get_yticklabels(), color='white')
        plt.tight_layout(); st.pyplot(fig); plt.close()
        
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        **{t('matrix_interpretation')}**
        - **{t('true_negatives')}:** {tn:,} - {t('tn_desc')}
        - **{t('false_positives')}:** {fp:,} - {t('fp_desc')}
        - **{t('false_negatives')}:** {fn:,} - {t('fn_desc')}
        - **{t('true_positives')}:** {tp:,} - {t('tp_desc')}
        """)
    
    with col2:
        st.subheader(t('feature_importance'))
        importance = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
        bars = ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors, edgecolor='white', linewidth=1)
        for bar, val in zip(bars, imp_df['Importance']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2., f'{val:.3f}',
                   ha='left', va='center', color='white', fontsize=10, fontweight='bold')
        ax.tick_params(colors='white', labelsize=10)
        ax.spines['bottom'].set_color('#333'); ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_xlim(0, max(importance) * 1.15)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        
        top = imp_df.iloc[-1]
        st.markdown(f"""
        **{t('key_insights')}**
        - **{t('most_important')}:** `{top['Feature']}` ({top['Importance']:.4f})
        - {t('importance_desc')}
        - {t('analysis_helps')}
        """)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.subheader(t('model_config'))
    with st.expander(t('view_params')):
        st.json({"Algorithm": "Random Forest", "Estimators": 100, "Max Depth": 20, "Features": feature_names})
    
    st.subheader(t('academic_refs'))
    st.info(f"""
    **{t('dataset_citation')}**
    Tavallaee, M., Bagheri, E., Lu, W., and Ghorbani, A. A. (2009).
    "A Detailed Analysis of the KDD CUP 99 Data Set."
    
    **{t('model_reference')}**
    Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
    """)

# =============================================================================
# APLIKASI UTAMA
# =============================================================================

def main():
    # Inisialisasi session state
    if 'lang' not in st.session_state:
        st.session_state.lang = 'id'
    if 'simulation_mode' not in st.session_state:
        st.session_state.simulation_mode = True
    if 'mikrotik_connected' not in st.session_state:
        st.session_state.mikrotik_connected = False
    
    lang = st.session_state.lang
    t = lambda k: get_text(k, lang)
    
    # Initialize MikroTik connection
    mikrotik_conn = MikroTikConnection(simulation_mode=st.session_state.simulation_mode)
    if st.session_state.simulation_mode:
        mikrotik_conn.connect()  # Auto-connect in simulation mode
    
    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            {get_icon('shield')}
            <h2 style="color: #00ffff; margin: 10px 0;">{t('app_title')}</h2>
            <p style="color: #888; font-size: 0.85rem;">Enterprise Edition v3.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Simulation Mode Toggle
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {get_icon('settings')}
            <span style="color: #00ffff; font-weight: 600;">Mode Operasi</span>
        </div>
        """, unsafe_allow_html=True)
        
        simulation_mode = st.checkbox(
            "🎮 Enable Simulation Mode", 
            value=st.session_state.simulation_mode,
            help="Centang untuk menggunakan data dummy (tanpa router fisik)"
        )
        st.session_state.simulation_mode = simulation_mode
        
        if simulation_mode:
            st.success("📡 Mode Simulasi Aktif")
        else:
            st.warning("🔌 Mode Real - Perlu Router MikroTik")
            # Show router login form
            st.markdown("**Konfigurasi Router:**")
            router_ip = st.text_input("IP Address", placeholder="192.168.88.1")
            router_user = st.text_input("Username", placeholder="admin")
            router_pass = st.text_input("Password", type="password")
            router_port = st.number_input("API Port", value=8728, min_value=1, max_value=65535)
            
            if st.button("🔗 Connect", use_container_width=True):
                mikrotik_conn = MikroTikConnection(simulation_mode=False)
                success, message = mikrotik_conn.connect(router_ip, router_user, router_pass, router_port)
                if success:
                    st.success(message)
                    st.session_state.mikrotik_connected = True
                else:
                    st.error(message)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Language Selection
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {get_icon('language')}
            <span style="color: #00ffff; font-weight: 600;">{t('select_language')}</span>
        </div>
        """, unsafe_allow_html=True)
        
        lang_choice = st.selectbox("", ["Bahasa Indonesia", "English"], 
                                   index=0 if lang == 'id' else 1, label_visibility="collapsed")
        st.session_state.lang = 'id' if lang_choice == "Bahasa Indonesia" else 'en'
        lang = st.session_state.lang
        t = lambda k: get_text(k, lang)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Navigation
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 15px;">
            {get_icon('dashboard')}
            <span style="color: #00ffff; font-weight: 600; font-size: 1.1rem;">{t('nav_title')}</span>
        </div>
        """, unsafe_allow_html=True)
        
        menu_options = {
            "soc": "🖥️ SOC Monitoring",
            "dashboard": t('menu_dashboard'),
            "simulation": t('menu_simulation'),
            "batch": t('menu_batch'),
            "performance": t('menu_performance')
        }
        
        selected = st.radio("", list(menu_options.values()), label_visibility="collapsed")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Status
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {get_icon('status')}
            <span style="color: #00ffff; font-weight: 600;">{t('system_status')}</span>
        </div>
        <div style="background: rgba(0, 255, 136, 0.1); border: 1px solid rgba(0, 255, 136, 0.3); 
                    border-radius: 8px; padding: 12px; margin: 8px 0; display: flex; align-items: center; gap: 10px;">
            {get_icon('activity')}
            <span style="color: #00ff88; font-weight: 500;">{t('ai_active')}</span>
        </div>
        <div style="background: rgba(0, 191, 255, 0.1); border: 1px solid rgba(0, 191, 255, 0.3); 
                    border-radius: 8px; padding: 12px; margin: 8px 0; text-align: center;">
            <span style="color: #00bfff;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p>{t('powered_by')}</p>
            <p>{t('copyright')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load Data & Model
    with st.spinner(t('init_error')):
        X_train, X_test, y_train, y_test, label_encoders, feature_names = load_and_preprocess_data()
        
        if X_train is None:
            st.error(f"""
            **{t('critical_error')}**
            
            Pastikan file dataset tersedia di: `{DATASET_PATH}`
            """)
            st.stop()
        
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
    
    # Page Routing
    if selected == "🖥️ SOC Monitoring":
        render_soc_monitoring_page(model, label_encoders, mikrotik_conn, lang)
    elif selected == t('menu_dashboard'):
        render_dashboard_page(model, metrics, X_test, y_test, lang)
    elif selected == t('menu_simulation'):
        render_simulation_page(model, label_encoders, feature_names, lang)
    elif selected == t('menu_batch'):
        render_batch_page(model, label_encoders, feature_names, lang)
    elif selected == t('menu_performance'):
        render_performance_page(model, metrics, feature_names, lang)

if __name__ == "__main__":
    main()
