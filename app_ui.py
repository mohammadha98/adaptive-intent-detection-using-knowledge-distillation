import streamlit as st
import json
import time
from src.orchestrator import Orchestrator
from src.utils import set_seed

# Initialize
set_seed()

st.set_page_config(page_title="AutoMTL: Adaptive Intent Detection", layout="wide")

# Session State for Orchestrator
if 'orchestrator' not in st.session_state:
    with st.spinner("Initializing System (Loading Models)..."):
        st.session_state.orchestrator = Orchestrator()

orchestrator = st.session_state.orchestrator


def save_entered_intents(intents_list):
    """Save intents list to entered_intents.json file."""
    try:
        with open("data/entered_intents.json", "w", encoding="utf-8") as f:
            json.dump(intents_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving intents to file: {e}")


def load_entered_intents():
    """Load intents from entered_intents.json file if exists."""
    import os
    intents_path = "data/entered_intents.json"
    if os.path.exists(intents_path):
        try:
            with open(intents_path, "r", encoding="utf-8") as f:
                loaded_intents = json.load(f)
                if isinstance(loaded_intents, list):
                    return loaded_intents
        except Exception as e:
            st.error(f"Error loading intents from file: {e}")
    return []

st.title("AutoMTL: Adaptive Self-Learning Intent Detection")
st.markdown("### MVP Sprint 1: Hybrid Student-Teacher Architecture")

# Tabs
tab1, tab2 = st.tabs(["Configuration (Phase 0)", "Live Inference"])

# --- TAB 1: Configuration ---
with tab1:
    st.header("Phase 0: Offline Warm-up")
    st.markdown("Define your intents and seed phrases. The **Teacher LLM** will augment this data, and the **Student Model** will train on it.")
    
    if 'intents_list' not in st.session_state:
        st.session_state.intents_list = load_entered_intents()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Current Intents")
        if st.session_state.intents_list:
            for item in st.session_state.intents_list:
                st.markdown(f"- {item['title']}: {', '.join(item.get('seeds', []))}")
        else:
            st.info("No intents added yet.")
        if st.button("Clear Intents"):
            st.session_state.intents_list = []
            save_entered_intents(st.session_state.intents_list)
            st.success("Cleared current intents.")

    with col_right:
        st.subheader("Add Intent (Form)")
        with st.form("add_intent_form"):
            title = st.text_input("Title")
            description = st.text_area("Description", height=120)
            seeds_csv = st.text_input("Seed Examples (comma-separated)")
            add_submitted = st.form_submit_button("Add Intent")
            if add_submitted:
                seeds = [s.strip() for s in seeds_csv.split(",") if s.strip()]
                if not title or not seeds:
                    st.warning("Title and at least one seed are required.")
                else:
                    st.session_state.intents_list.append({
                        "title": title,
                        "description": description,
                        "seeds": seeds
                    })
                    st.success(f"Added intent '{title}'")
                    # Save intents to file
                    save_entered_intents(st.session_state.intents_list)

        st.divider()
        st.subheader("Add Intents (JSON)")
        st.markdown("Paste a JSON list of intent objects to bulk add.")
        
        json_input = st.text_area("JSON Input", height=150, placeholder='[{"title": "...", "description": "...", "seeds": ["..."]}]')
        if st.button("Add from JSON"):
            try:
                new_intents = json.loads(json_input)
                if isinstance(new_intents, list):
                    count = 0
                    for item in new_intents:
                        if "title" in item and "seeds" in item:
                            st.session_state.intents_list.append(item)
                            count += 1
                    if count > 0:
                        st.success(f"Successfully added {count} intents from JSON.")
                        # Save intents to file
                        save_entered_intents(st.session_state.intents_list)
                    else:
                        st.warning("No valid intents found in JSON list.")
                else:
                    st.error("JSON must be a list of objects.")
            except json.JSONDecodeError:
                st.error("Invalid JSON format.")
            except Exception as e:
                st.error(f"Error processing JSON: {e}")

    with st.form("bootstrap_form"):
        submit_bootstrap = st.form_submit_button("Bootstrap Model")
        if submit_bootstrap:
            try:
                if not st.session_state.intents_list:
                    st.warning("Add at least one intent before bootstrapping.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Teacher Agent: Generating synthetic data...")
                    progress_bar.progress(20)
                    with st.spinner("Processing..."):
                        count = orchestrator.bootstrap_system(st.session_state.intents_list)
                    progress_bar.progress(100)
                    status_text.text("Done!")
                    st.success(f"Bootstrap Complete! Generated and trained on {count} samples.")
            except Exception as e:
                st.error(f"Error during bootstrap: {e}")

# --- TAB 2: Live Inference ---
with tab2:
    st.header("Live Inference")
    
    # --- Sidebar for Active Learning ---
    with st.sidebar:
        st.header("Active Learning Loop")
        st.info("When the Teacher takes over, data is saved. Retrain the Student to learn from these edge cases.")
        
        # # We can't easily get real-time file updates in Streamlit without rerun, 
        # # so we'll just check on load or button press.
        # feedback_data = orchestrator.feedback_manager.get_feedback_data()
        # st.metric("New Feedback Samples", len(feedback_data))
        
        if st.button("Retrain Student (Active Learning)"):
            with st.spinner("Retraining on combined dataset..."):
                count = orchestrator.retrain_student()
            st.success(f"Retraining Complete! Total Samples: {count}")
            time.sleep(1)
            st.rerun()

    st.markdown("Test the system. Queries are first sent to the **Student**. If confidence is low, they fallback to the **Teacher**.")
    
    query = st.text_input("Enter User Query:", placeholder="e.g., I want to send 50 bucks")
    
    if st.button("Predict"):
        if not query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Analyzing..."):
                result = orchestrator.handle_query(query)
            
            st.subheader("Result Analysis")
            
            # Display Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Label", result['label'])
            with col2:
                st.metric("Confidence", f"{result['confidence']:.2%}")
            with col3:
                source = result['source'].upper()
                # Streamlit doesn't support :color[] in metric label easily, using markdown
                st.markdown(f"**Source**")
                if result.get('source', '').lower() == 'student':
                    st.success(source)
                else:
                    st.warning(source)
            
            # Visual Flow Explanation
            st.divider()
            if result['source'] == 'student':
                st.info(f"✅ **Student Model Success**: The model was confident (> 75%) that this is '{result['label']}'.")
            else:
                st.warning(f"⚠️ **Student Uncertain**: Confidence was low. Routed to **Teacher LLM** for fallback prediction.")

            # JSON Debug
            with st.expander("Debug Details"):
                st.json(result)
