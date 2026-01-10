import streamlit as st
import os
import yaml
from utils.workspace_manager import WorkspaceManager

# Page configuration
st.set_page_config(
    page_title="scRNA-seq Explorer",
    layout="wide",
    page_icon="png/sidebar_logo.png",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'workspace' not in st.session_state:
        st.session_state.workspace = None
    if 'adata' not in st.session_state:
        st.session_state.adata = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "workspace"
    if 'workspace_manager' not in st.session_state:
        st.session_state.workspace_manager = WorkspaceManager(config['workspace']['base_dir'])

init_session_state()

# Custom CSS
def load_dark_theme():
    css = """
    <style>
    
        /* ------------------ Fonts ------------------ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
        color: #fafafa !important;
        font-weight: 700 !important;
        transition: background 0.3s, color 0.3s;
    }

    /* Explicitly target text elements for maximum visibility */
    p, h1, h2, h3, h4, h5, h6, li, span, label, .stMarkdown, .stText {
        color: #fafafa !important;
        font-weight: 700 !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background: #262730;
        padding: 2px 4px;
        border-radius: 4px;
    }

    /* ------------------ App Background ------------------ */
    .stApp {
        background: #0e1117;
        min-height: 100vh;
        transition: background 0.5s ease;
    }

    /* ------------------ Header ------------------ */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #0066cc, #8a2be2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease forwards;
    }

    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* ------------------ Cards ------------------ */
    .card {
        background: #1e1e1e;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #333333;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        backdrop-filter: blur(12px);
        transition: all 0.4s ease;
    }

    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,102,204,0.15);
        border-color: #0066cc;
    }

    /* ------------------ Metric Cards ------------------ */
    .metric-card {
        background: #262730;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #333333;
        transition: all 0.4s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .metric-card:hover {
        border-color: #0066cc;
        transform: scale(1.05) translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,102,204,0.15);
    }

    /* ------------------ Buttons ------------------ */
    .stButton>button {
        border-radius: 14px;
        border: 2px solid #0066cc;
        background: #1e1e1e;
        color: #0066cc;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.4s ease;
        box-shadow: 0 2px 8px rgba(0,102,204,0.1);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #0066cc, #8a2be2);
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,102,204,0.3);
    }

    /* ------------------ Progress Bars ------------------ */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #0066cc, #8a2be2);
        transition: all 0.4s ease;
    }

    /* ------------------ Scrollbar ------------------ */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }

    ::-webkit-scrollbar-thumb {
        background: #0066cc;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #0052a3;
    }

    /* ------------------ Animations ------------------ */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeInUp 1s ease forwards;
    }

    /* ------------------ Sidebar ------------------ */
    [data-testid="stSidebar"] {
        background: #262730;
        padding: 1rem;
        border-right: 1px solid #333333;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {
        color: #fafafa !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] h2 {
        font-weight: 700;
        color: #0066cc;
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        margin-bottom: 0.8rem;
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_dark_theme()


# Main app
def main():
    # Center the header logo using columns
    _, col2, _ = st.columns([1, 4, 1])
    with col2:
        st.video("png/video.mp4", autoplay=True, loop=True, muted=True)

    
    # Sidebar for workspace info and navigation
    with st.sidebar:
        
        st.header("🔬 Workspace")
        
        if st.session_state.workspace:
            st.success(f"Active: {st.session_state.workspace}")
            
            # Show dataset info if available
            if st.session_state.adata is not None:
                st.info(f"Cells: {st.session_state.adata.n_obs:,} | Genes: {st.session_state.adata.n_vars:,}")
            
            st.divider()
        
        st.header("Analysis Steps")
        
        # Quick actions
        st.header("Quick Actions")
        if st.button("Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()
        
        if st.button("Reload Config", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    # Main content area
    # Centered welcome header
    st.markdown("<h2 style='text-align: center;'>Welcome to the scRNA-seq Analysis Platform!</h2>", unsafe_allow_html=True)
    st.markdown("""   


    This application provides a complete workflow for single-cell RNA-seq data analysis using standard methods.
    
    ### Getting Started:
    1. **Create or Load** a workspace from the Workspace page
    2. **Upload** your single-cell data
    3. Follow the step-by-step analysis workflow
    4. **Export** your results and reproducible code
    
    ### Supported Data Formats:
    - AnnData (.h5ad)
    - Loom (.loom) 
    - 10X Genomics (mtx + genes.tsv + barcodes.tsv)
    - CSV/TSV count matrices
    
    Navigate through the pages using the sidebar to begin your analysis!
                

            
    """)
    
    # Display current status
    col1, col2, col3 = st.columns(3)
    
    # Use the .metric-card style for the status display
    with col1:
        with st.container(border=True):
            st.metric("Workspace", st.session_state.workspace or "Not set")
            
    with col2:
        with st.container(border=True):
            data_status = "Loaded" if st.session_state.adata is not None else "Not loaded"
            st.metric("Data", data_status)
            
    with col3:
        with st.container(border=True):
            st.metric("Analysis Steps", "0/8" if st.session_state.adata is None else "1/8")
            

            
if __name__ == "__main__":
    main()