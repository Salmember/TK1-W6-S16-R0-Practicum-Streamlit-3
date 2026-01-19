import streamlit as st
import numpy as np
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="CNN Image Classifier - Premium Edition",
    page_icon="👑",
    layout="wide"
)

# Premium Custom CSS with Gold & Silver Accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.5); }
        50% { box-shadow: 0 0 40px rgba(255, 215, 0, 0.8); }
    }
    
    .main {
        background: linear-gradient(135deg, #FFF8DC 0%, #FFFACD 25%, #F0F0F0 50%, #FFFACD 75%, #FFF8DC 100%);
    }
    
    .premium-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.2), transparent);
        background-size: 200% 100%;
        animation: shimmer 3s infinite;
    }
    
    .gold-text {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
    }
    
    .silver-text {
        background: linear-gradient(135deg, #E8E8E8 0%, #C0C0C0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .premium-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 2px solid #FFD700;
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    .silver-card {
        background: linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px solid #C0C0C0;
        box-shadow: 0 6px 24px rgba(192, 192, 192, 0.2);
    }
    
    .gold-badge {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1a1a1a;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 165, 0, 0.4);
        animation: glow 2s ease-in-out infinite;
    }
    
    .silver-badge {
        background: linear-gradient(135deg, #E8E8E8 0%, #C0C0C0 100%);
        color: #1a1a1a;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(192, 192, 192, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.95);
        padding: 10px;
        border-radius: 15px;
        border: 2px solid #FFD700;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: white;
        border-radius: 12px;
        padding: 0 30px;
        font-weight: 700;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1a1a1a;
        border: 2px solid #FFD700;
        box-shadow: 0 4px 15px rgba(255, 165, 0, 0.4);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1a1a1a;
        font-weight: 800;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        box-shadow: 0 6px 20px rgba(255, 165, 0, 0.4);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(255, 165, 0, 0.6);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
    }
    
    .premium-metric {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: #1a1a1a;
        box-shadow: 0 8px 32px rgba(255, 165, 0, 0.4);
        animation: glow 2s ease-in-out infinite;
    }
    
    .silver-metric {
        background: linear-gradient(135deg, #E8E8E8 0%, #C0C0C0 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: #1a1a1a;
        box-shadow: 0 8px 32px rgba(192, 192, 192, 0.3);
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
    }
    
    h1, h2, h3 {
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CLASS_LABELS = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

ARCHITECTURES = {
    'custom': {
        'name': '⭐ Custom CNN',
        'layers': ['Conv2D(32)', 'BatchNorm', 'MaxPool', 'Conv2D(64)', 
                   'BatchNorm', 'MaxPool', 'Dense(128)', 'Dropout(0.5)', 'Dense(10)'],
        'params': '~150K',
        'accuracy': '94.2%'
    },
    'resnet': {
        'name': '👑 ResNet50 (Elite)',
        'layers': ['ResNet50(Pretrained)', 'GlobalAvgPool', 'Dense(256)', 
                   'Dropout(0.3)', 'Dense(10)'],
        'params': '~25M',
        'accuracy': '97.8%'
    },
    'vgg': {
        'name': '💎 VGG16 (Premium)',
        'layers': ['VGG16(pretrained)', 'Flatten', 'Dense(512)', 
                   'Dropout(0.5)', 'Dense(10)'],
        'params': '~15M',
        'accuracy': '96.5%'
    }
}

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Premium Header
st.markdown("""
<div class="premium-header" style="position: relative; z-index: 1;">
    <div style="display: flex; justify-content: space-between; align-items: center; position: relative; z-index: 2;">
        <div>
            <div class="gold-text">👑 CNN IMAGE CLASSIFIER</div>
            <p class="silver-text" style="font-size: 1.2rem; margin-top: 0.5rem;">Premium Deep Learning Inference System</p>
        </div>
        <div style="display: flex; gap: 1rem;">
            <div class="gold-badge">⚡ ELITE MODEL</div>
            <div class="silver-badge">✓ READY</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Premium Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 UPLOAD & CLASSIFY", 
    "⚙️ ELITE CONFIGURATION", 
    "🏗️ ARCHITECTURE", 
    "📊 PERFORMANCE"
])

# Tab 1: Upload & Classify
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ PREMIUM IMAGE UPLOAD")
        
        uploaded_file = st.file_uploader(
            "Select Elite Image File", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload premium image for classification (max 5MB)"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="✨ Elite Image Loaded", use_container_width=True)
            
            st.markdown("---")
            arch_choice = st.selectbox(
                "SELECT ARCHITECTURE",
                options=['custom', 'resnet', 'vgg'],
                format_func=lambda x: ARCHITECTURES[x]['name'],
                key='inference_arch'
            )
            
            if st.button("RUN ELITE CLASSIFICATION", type="primary", use_container_width=True):
                with st.spinner("⚡ Processing with Elite Neural Networks..."):
                    progress_bar = st.progress(0, text="Initializing elite model...")
                    time.sleep(0.3)
                    
                    progress_bar.progress(25, text="Preprocessing image...")
                    time.sleep(0.3)
                    
                    progress_bar.progress(50, text="Running inference...")
                    time.sleep(0.4)
                    
                    progress_bar.progress(75, text="Analyzing predictions...")
                    time.sleep(0.3)
                    
                    progress_bar.progress(100, text="Complete!")
                    time.sleep(0.2)
                    
                    # Generate elite predictions
                    base_accuracy = {
                        'custom': 0.88,
                        'resnet': 0.95,
                        'vgg': 0.92
                    }[arch_choice]
                    
                    top_idx = np.random.randint(0, len(CLASS_LABELS))
                    top_conf = base_accuracy + np.random.uniform(-0.02, 0.05)
                    
                    confidences = np.random.random(len(CLASS_LABELS)) * 0.3 * (1 - top_conf)
                    confidences[top_idx] = top_conf
                    
                    sorted_indices = np.argsort(confidences)[::-1]
                    
                    st.session_state.predictions = {
                        'results': [
                            {'class': CLASS_LABELS[i], 'confidence': confidences[i]}
                            for i in sorted_indices[:5]
                        ],
                        'inference_time': f"{np.random.uniform(50, 150):.2f}ms",
                        'preprocess_time': "23.5ms",
                        'architecture': ARCHITECTURES[arch_choice]['name']
                    }
                    
                    st.balloons()
                    st.success("✅ Elite Classification Complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="silver-card">', unsafe_allow_html=True)
        st.markdown("### 👑ELITE PREDICTION RESULTS")
        
        if st.session_state.predictions:
            pred = st.session_state.predictions
            
            # Top prediction
            st.markdown(f"""
            <div class="premium-metric">
                <div style="font-size: 0.9rem; font-weight: 700; opacity: 0.9;">👑 TOP PREDICTION</div>
                <div style="font-size: 2.5rem; font-weight: 900; margin: 1rem 0;">
                    {pred['results'][0]['class']}
                </div>
                <div style="font-size: 3.5rem; font-weight: 900;">
                    {pred['results'][0]['confidence']*100:.2f}%
                </div>
                <div style="font-size: 0.9rem; font-weight: 700; margin-top: 1rem; opacity: 0.9;">
                    {pred['architecture']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🏆 TOP 5 ELITE PREDICTIONS**")
            
            icons = ['👑', '🥈', '🥉', '📊', '📊']
            for i, result in enumerate(pred['results']):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{icons[i]} {result['class']}**")
                with col_b:
                    st.markdown(f"**{result['confidence']*100:.2f}%**")
                st.progress(result['confidence'])
            
            # Performance metrics
            st.markdown("---")
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); border-radius: 12px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #1a1a1a;">⚡ PREPROCESSING</div>
                    <div style="font-size: 1.5rem; font-weight: 900; color: #1a1a1a;">{pred['preprocess_time']}</div>
                </div>
                """, unsafe_allow_html=True)
            with metric_col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #E8E8E8 0%, #C0C0C0 100%); border-radius: 12px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #1a1a1a;">INFERENCE</div>
                    <div style="font-size: 1.5rem; font-weight: 900; color: #1a1a1a;">{pred['inference_time']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 0; color: #999;">
                <div style="font-size: 5rem;">⚠️</div>
                <p style="font-size: 1.2rem; font-weight: 700; margin-top: 1rem;">Upload Elite Image</p>
                <p style="color: #666;">To see premium predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Elite Configuration
with tab2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ ELITE MODEL CONFIGURATION")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="silver-card">', unsafe_allow_html=True)
        architecture = st.selectbox(
            "🏗️ CNN ARCHITECTURE",
            options=['custom', 'resnet', 'vgg'],
            format_func=lambda x: f"{ARCHITECTURES[x]['name']} ({ARCHITECTURES[x]['params']})"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="silver-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        batch_norm = st.checkbox(
            "✨ BATCH NORMALIZATION",
            value=True,
            help="Elite convergence acceleration"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="silver-card">', unsafe_allow_html=True)
        dropout = st.slider(
            "DROPOUT RATE",
            min_value=0.0,
            max_value=0.7,
            value=0.5,
            step=0.1,
            help="Premium regularization technique"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="silver-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        transfer_learning = st.checkbox(
            "🎖️ TRANSFER LEARNING",
            value=False,
            help="Use premium ImageNet weights"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 ELITE TRAINING HYPERPARAMETERS")
    
    col1, col2, col3, col4 = st.columns(4)
    metrics_style = "text-align: center; padding: 1.5rem; background: white; border-radius: 12px; border: 2px solid #FFD700;"
    
    with col1:
        st.markdown(f'<div style="{metrics_style}"><div style="font-size: 0.8rem; font-weight: 700; color: #666;">LEARNING RATE</div><div style="font-size: 1.5rem; font-weight: 900; color: #1a1a1a;">0.001</div><div style="font-size: 0.7rem; color: #888;">Adam</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="{metrics_style}"><div style="font-size: 0.8rem; font-weight: 700; color: #666;">BATCH SIZE</div><div style="font-size: 1.5rem; font-weight: 900; color: #1a1a1a;">32</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div style="{metrics_style}"><div style="font-size: 0.8rem; font-weight: 700; color: #666;">EPOCHS</div><div style="font-size: 1.5rem; font-weight: 900; color: #1a1a1a;">50</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div style="{metrics_style}"><div style="font-size: 0.8rem; font-weight: 700; color: #666;">OPTIMIZER</div><div style="font-size: 1.5rem; font-weight: 900; color: #1a1a1a;">ADAM</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Architecture
with tab3:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 🏗️ ELITE NETWORK ARCHITECTURES")
    
    for key, arch in ARCHITECTURES.items():
        with st.expander(f"**{arch['name']}** • {arch['params']} • {arch['accuracy']}", expanded=True):
            for i, layer in enumerate(arch['layers'], 1):
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 0.75rem 0;">
                    <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
                                border-radius: 10px; color: #1a1a1a; display: flex; align-items: center; 
                                justify-content: center; font-weight: 900; margin-right: 1rem; box-shadow: 0 4px 12px rgba(255, 165, 0, 0.3);">
                        {i}
                    </div>
                    <div style="flex: 1; padding: 1rem; background: white; 
                                border: 2px solid #e0e0e0; border-radius: 12px; transition: all 0.3s;">
                        <code style="font-weight: 700; color: #1a1a1a;">{layer}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Performance
with tab4:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📊 ELITE PERFORMANCE COMPARISON")
    
    import pandas as pd
    
    perf_data = {
        'ARCHITECTURE': ['⭐ Custom CNN', '👑 ResNet50', '💎 VGG16'],
        'ACCURACY': ['94.2%', '97.8%', '96.5%'],
        'PARAMETERS': ['~150K', '~25M', '~15M'],
        'INFERENCE': ['~45ms', '~120ms', '~95ms']
    }
    
    df = pd.DataFrame(perf_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Premium Metric Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="premium-metric">
            <div style="font-size: 0.9rem; font-weight: 700; opacity: 0.9;">TRAINING ACCURACY</div>
            <div style="font-size: 3.5rem; font-weight: 900; margin: 1rem 0;">96.8%</div>
            <div style="font-size: 0.8rem; font-weight: 700; opacity: 0.9;">After 50 Elite Epochs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="silver-metric">
            <div style="font-size: 0.9rem; font-weight: 700; opacity: 0.9;">✅ VALIDATION ACCURACY</div>
            <div style="font-size: 3.5rem; font-weight: 900; margin: 1rem 0;">94.2%</div>
            <div style="font-size: 0.8rem; font-weight: 700; opacity: 0.9;">Held-out Premium Set</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #FFD700 0%, #C0C0C0 50%, #FFD700 100%); 
                    padding: 2rem; border-radius: 20px; text-align: center; color: #1a1a1a;
                    box-shadow: 0 8px 32px rgba(255, 165, 0, 0.4);">
            <div style="font-size: 0.9rem; font-weight: 700; opacity: 0.9;">⚡ AVG INFERENCE</div>
            <div style="font-size: 3.5rem; font-weight: 900; margin: 1rem 0;">75ms</div>
            <div style="font-size: 0.8rem; font-weight: 700; opacity: 0.9;">Per Premium Image</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Elite Techniques
    st.markdown('<div class="silver-card">', unsafe_allow_html=True)
    st.markdown("### 💡 ELITE TECHNIQUES & IMPACT")
    
    techniques = [
        ("🔵", "Dropout Regularization", "Reduced overfitting by 8.3%, improved validation accuracy by 2.1%"),
        ("🟢", "Batch Normalization", "Accelerated training by 35%, stabilized gradient flow"),
        ("🟣", "Transfer Learning", "Boosted accuracy by 3.6% using pretrained ImageNet weights"),
        ("🟠", "Adam Optimizer", "Converged 2.5x faster than SGD with momentum")
    ]
    
    for emoji, title, desc in techniques:
        st.markdown(f"""
        <div style="display: flex; align-items: start; margin: 1.5rem 0; padding: 1rem; 
                    background: white; border-radius: 12px; border: 2px solid #f0f0f0;">
            <div style="font-size: 2rem; margin-right: 1rem;">{emoji}</div>
            <div>
                <div style="font-weight: 800; font-size: 1.1rem; margin-bottom: 0.5rem;">{title}</div>
                <div style="color: #666; font-weight: 500;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Premium Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
            padding: 2rem; border-radius: 20px; box-shadow: 0 8px 32px rgba(255, 165, 0, 0.4);">
    <div style="display: flex; align-items: center;">
        <div style="font-size: 3rem; margin-right: 1.5rem;">ℹ️</div>
        <div>
            <div style="font-weight: 900; font-size: 1.3rem; color: #1a1a1a; margin-bottom: 0.5rem;">
                🌟 PREMIUM DEMO MODE
            </div>
            <div style="color: #1a1a1a; font-weight: 600;">
                This elite simulation demonstrates advanced CNN architecture and inference workflow. 
                For production deployment, integrate with TensorFlow/PyTorch on premium cloud infrastructure.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)