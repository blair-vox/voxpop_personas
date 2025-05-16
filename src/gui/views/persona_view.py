"""
View component for persona generation.
"""
import streamlit as st
from typing import List
from models.persona import PersonaResponse

def render_persona_generation(controller) -> None:
    """Render the persona generation interface."""
    st.header("Persona Generation")
    
    # Number of personas to generate
    num_personas = st.number_input(
        "Number of personas to generate",
        min_value=1,
        max_value=100,
        value=5
    )
    
    # Generate button
    if st.button("Generate Personas"):
        with st.spinner("Generating personas..."):
            responses = controller.generate_personas(num_personas)
            controller.current_responses = responses
            
            # Display results
            st.success(f"Generated {len(responses)} personas!")
            
            # Show each persona
            for i, response in enumerate(responses, 1):
                with st.expander(f"Persona {i}"):
                    # Persona details
                    st.subheader("Demographics")
                    for key, value in response.persona_details.items():
                        st.write(f"**{key}**: {value}")
                    
                    # Narrative response
                    st.subheader("Narrative Response")
                    st.write(response.narrative_response)
                    
                    # Survey response
                    st.subheader("Survey Response")
                    st.write(f"Support Level: {response.survey_response.support_level}")
                    st.write(f"Impact on Housing: {response.survey_response.impact_on_housing}")
                    st.write(f"Impact on Transport: {response.survey_response.impact_on_transport}")
                    st.write(f"Impact on Community: {response.survey_response.impact_on_community}")
                    
                    # Key concerns
                    st.write("Key Concerns:")
                    for concern in response.survey_response.key_concerns:
                        st.write(f"- {concern}")
                    
                    # Suggested improvements
                    st.write("Suggested Improvements:")
                    for improvement in response.survey_response.suggested_improvements:
                        st.write(f"- {improvement}")
                    
                    # Themes
                    st.subheader("Themes")
                    st.write("Key Themes:")
                    for theme in response.key_themes:
                        st.write(f"- {theme}")
                    
                    if response.canonical_themes:
                        st.write("Canonical Themes:")
                        for theme, confidence in response.canonical_themes.items():
                            st.write(f"- {theme} (confidence: {confidence:.2f})")
    
    # Save button
    if controller.current_responses:
        if st.button("Save Responses"):
            output_file = controller.save_responses()
            if output_file:
                st.success(f"Responses saved to: {output_file}")
            else:
                st.error("Failed to save responses") 