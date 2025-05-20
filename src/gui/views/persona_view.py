"""
View component for persona generation.
"""
import streamlit as st
from typing import List
from models.persona import PersonaResponse
import os
import glob
import json
from config.prompts.persona_prompts import PERSONA_TEMPLATE

def render_persona_generation(controller) -> None:
    """Render the persona generation interface."""
    st.header("Persona Generation")
    
    # Question Editor (Local Proposal)
    st.subheader("Local Proposal / Question")
    st.write("Edit the local proposal or question that will be posed to the LLM along with persona details:")
    default_question = (
        "Waverley Council is considering a policy that would remove minimum parking requirements for new apartment developments in Bondi. "
        "This means developers could build fewer or no car spaces if they believe it suits the residents' needs."
    )
    if 'llm_question' not in st.session_state:
        st.session_state.llm_question = default_question
    edited_question = st.text_area(
        "Local Proposal / Question",
        value=st.session_state.llm_question,
        height=100
    )
    if edited_question != st.session_state.llm_question:
        st.session_state.llm_question = edited_question
        st.info("Question updated. This will be used for the next persona generation.")

    # LLM Prompt Editor
    st.subheader("LLM Prompt")
    st.write("Edit the prompt that will be used to generate persona responses:")
    if 'llm_prompt' not in st.session_state:
        st.session_state.llm_prompt = PERSONA_TEMPLATE
    edited_prompt = st.text_area(
        "LLM Prompt",
        value=st.session_state.llm_prompt,
        height=400
    )
    if edited_prompt != st.session_state.llm_prompt:
        st.session_state.llm_prompt = edited_prompt
        st.info("Prompt updated. This will be used for the next persona generation.")
    
    # Number of personas to generate
    num_personas = st.number_input(
        "Number of personas to generate",
        min_value=1,
        max_value=100,
        value=5
    )
    
    # Random state option
    use_random_state = st.checkbox("Use fixed random state for reproducibility", value=False)
    random_state = None
    if use_random_state:
        random_state = st.number_input(
            "Random state (integer)",
            min_value=0,
            max_value=1000000,
            value=42
        )
    
    # Generate button
    if st.button("Generate Personas"):
        with st.spinner("Generating personas..."):
            responses = controller.generate_personas(
                num_personas,
                random_state=random_state,
                custom_prompt=st.session_state.llm_prompt,
                question=st.session_state.llm_question
            )
            controller.current_responses = responses
            
            # Display results
            st.success(f"Generated {len(responses)} personas!")
            
            # Show each persona
            for i, response in enumerate(responses, 1):
                with st.expander(f"Persona {i}"):
                    # Persona details
                    st.subheader("Demographics")
                    print("[DEBUG] persona_details for Persona", i, ":", response.persona_details)
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
                        if isinstance(response.canonical_themes, dict):
                            for theme, confidence in response.canonical_themes.items():
                                st.write(f"- {theme} (confidence: {confidence:.2f})")
                        elif isinstance(response.canonical_themes, list):
                            for theme in response.canonical_themes:
                                st.write(f"- {theme}")
                        else:
                            st.write(response.canonical_themes)
    
    # Save button
    if controller.current_responses:
        if st.button("Save Responses"):
            output_file = controller.save_responses()
            if output_file:
                st.success(f"Responses saved to: {output_file}")
            else:
                st.error("Failed to save responses")

    # Load Most Recent Responses button
    if st.button("Load Most Recent Responses"):
        output_dir = "output"
        files = glob.glob(os.path.join(output_dir, "all_responses_*.json"))
        if files:
            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, "r") as f:
                data = json.load(f)
            responses = []
            for item in data:
                if isinstance(item, dict) and not isinstance(item, PersonaResponse):
                    try:
                        responses.append(PersonaResponse(**item))
                    except Exception as e:
                        st.warning(f"Could not parse a response: {e}")
                elif isinstance(item, PersonaResponse):
                    responses.append(item)
            controller.current_responses = responses
            st.session_state['show_loaded'] = True
            st.success(f"Loaded {len(responses)} responses from {os.path.basename(latest_file)}")
            st.rerun()
        else:
            st.warning("No saved responses found in output directory.")

    # Always show current responses if present
    if controller.current_responses and not st.session_state.get('show_loaded', False):
        st.info(f"Currently displaying {len(controller.current_responses)} personas.")
        for i, response in enumerate(controller.current_responses, 1):
            persona_name = response.persona_details.get('name', f"Persona_{i}")
            with st.expander(f"Persona {i} ({persona_name})"):
                st.subheader("Demographics")
                print("[DEBUG] persona_details for Persona", i, ":", response.persona_details)
                for key, value in response.persona_details.items():
                    st.write(f"**{key}**: {value}")
                st.subheader("Narrative Response")
                st.write(response.narrative_response)
                st.subheader("Survey Response")
                st.write(f"Support Level: {response.survey_response.support_level}")
                st.write(f"Impact on Housing: {response.survey_response.impact_on_housing}")
                st.write(f"Impact on Transport: {response.survey_response.impact_on_transport}")
                st.write(f"Impact on Community: {response.survey_response.impact_on_community}")
                st.write("Key Concerns:")
                for concern in response.survey_response.key_concerns:
                    st.write(f"- {concern}")
                st.write("Suggested Improvements:")
                for improvement in response.survey_response.suggested_improvements:
                    st.write(f"- {improvement}")
                st.subheader("Themes")
                st.write("Key Themes:")
                for theme in response.key_themes:
                    st.write(f"- {theme}")
                if response.canonical_themes:
                    st.write("Canonical Themes:")
                    if isinstance(response.canonical_themes, dict):
                        for theme, confidence in response.canonical_themes.items():
                            st.write(f"- {theme} (confidence: {confidence:.2f})")
                    elif isinstance(response.canonical_themes, list):
                        for theme in response.canonical_themes:
                            st.write(f"- {theme}")
                    else:
                        st.write(response.canonical_themes)
        st.session_state['show_loaded'] = False

    # Show loaded personas if flag is set
    if st.session_state.get('show_loaded', False) and controller.current_responses:
        st.success(f"Loaded {len(controller.current_responses)} personas from file.")
        for i, response in enumerate(controller.current_responses, 1):
            persona_name = response.persona_details.get('name', f"Persona_{i}")
            with st.expander(f"Persona {i} ({persona_name})"):
                st.subheader("Demographics")
                print("[DEBUG] persona_details for Persona", i, ":", response.persona_details)
                for key, value in response.persona_details.items():
                    st.write(f"**{key}**: {value}")
                st.subheader("Narrative Response")
                st.write(response.narrative_response)
                st.subheader("Survey Response")
                st.write(f"Support Level: {response.survey_response.support_level}")
                st.write(f"Impact on Housing: {response.survey_response.impact_on_housing}")
                st.write(f"Impact on Transport: {response.survey_response.impact_on_transport}")
                st.write(f"Impact on Community: {response.survey_response.impact_on_community}")
                st.write("Key Concerns:")
                for concern in response.survey_response.key_concerns:
                    st.write(f"- {concern}")
                st.write("Suggested Improvements:")
                for improvement in response.survey_response.suggested_improvements:
                    st.write(f"- {improvement}")
                st.subheader("Themes")
                st.write("Key Themes:")
                for theme in response.key_themes:
                    st.write(f"- {theme}")
                if response.canonical_themes:
                    st.write("Canonical Themes:")
                    if isinstance(response.canonical_themes, dict):
                        for theme, confidence in response.canonical_themes.items():
                            st.write(f"- {theme} (confidence: {confidence:.2f})")
                    elif isinstance(response.canonical_themes, list):
                        for theme in response.canonical_themes:
                            st.write(f"- {theme}")
                    else:
                        st.write(response.canonical_themes) 