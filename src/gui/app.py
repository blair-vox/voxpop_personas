"""
Main Streamlit application for VoxPop Personas.
"""
import streamlit as st
from pathlib import Path
import sys

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from gui.controllers.persona_controller import PersonaController
from gui.views.persona_view import render_persona_generation
from gui.views.analysis_view import render_analysis
from gui.views.theme_view import render_theme_analysis

def main():
    st.set_page_config(
        page_title="VoxPop Personas",
        page_icon="👥",
        layout="wide"
    )
    
    st.title("VoxPop Personas")
    
    # Initialize controller
    if 'controller' not in st.session_state:
        st.session_state['controller'] = PersonaController()
    controller = st.session_state['controller']
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Persona Generation", "Theme Analysis", "Data Analysis"]
    )
    
    # Main content area
    if page == "Persona Generation":
        render_persona_generation(controller)
    elif page == "Theme Analysis":
        render_theme_analysis(controller)
    else:
        render_analysis(controller)

if __name__ == "__main__":
    main() 