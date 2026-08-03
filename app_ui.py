import streamlit as st
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from src.orchestrator import Orchestrator
from src.utils import set_seed
from src.evaluate import run_evaluation, AutoMTLEvaluator

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
tab1, tab2, tab3 = st.tabs(["Configuration (Phase 0)", "Live Inference", "Evaluation"])

# --- TAB 1: Configuration ---
with tab1:
    st.header("Phase 0: Offline Warm-up")
    st.markdown("Define your intents and seed phrases. The **Teacher LLM** will augment this data, and the **Student Model** will train on it.")
    
    # Bootstrap Configuration
    st.subheader("Bootstrap Configuration")
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        bootstrap_mode = st.radio(
            "Bootstrap Mode:",
            ["Run without Dataset (Standard/Mock)", "Run with CLINC150 Dataset"],
            help="Choose whether to bootstrap with CLINC150 dataset or start from scratch"
        )
    
    with col_config2:
        if bootstrap_mode == "Run with CLINC150 Dataset":
            use_data_augmentation = st.checkbox("Enable Data Augmentation", value=False,
                                              help="Enable synthetic data generation (recommended to disable for evaluation)")
        else:
            use_data_augmentation = st.checkbox("Enable Data Augmentation", value=True,
                                              help="Enable synthetic data generation for mock mode")
            st.info("Mock mode: System will start without pre-defined dataset")
    
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
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if bootstrap_mode == "Run with CLINC150 Dataset":
                    # Bootstrap با دیتاست CLINC150
                    status_text.text("Loading CLINC150 Dataset...")
                    progress_bar.progress(10)
                    
                    with st.spinner("Processing CLINC150 dataset..."):
                        count = orchestrator.bootstrap_system(
                            intents=None,  # استفاده از دیتاست
                            use_clinc_dataset=True,
                            dataset_path="datasets/clinic150/data/data_full.json",
                            enable_augmentation=use_data_augmentation,
                            max_samples_per_intent=100  # محدود کردن برای شروع سریع
                        )
                    
                    progress_bar.progress(100)
                    status_text.text("Done!")
                    if count > 0:
                        st.success(f"Bootstrap Complete! Loaded and trained on {count} CLINC150 samples.")
                    else:
                        st.error("Failed to load CLINC150 dataset. Please check if the dataset exists.")
                        
                else:
                    # Bootstrap استاندارد (Mock mode)
                    if not st.session_state.intents_list:
                        st.warning("Add at least one intent before bootstrapping in mock mode.")
                    else:
                        status_text.text("Teacher Agent: Generating synthetic data...")
                        progress_bar.progress(20)
                        
                        with st.spinner("Processing..."):
                            count = orchestrator.bootstrap_system(
                                intents=st.session_state.intents_list,
                                use_clinc_dataset=False,
                                enable_augmentation=use_data_augmentation
                            )
                        
                        progress_bar.progress(100)
                        status_text.text("Done!")
                        st.success(f"Bootstrap Complete! Generated and trained on {count} samples.")
                        
            except Exception as e:
                st.error(f"Error during bootstrap: {e}")
                import traceback
                traceback.print_exc()

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

# --- TAB 3: Evaluation ---
with tab3:
    st.header("System Evaluation with CLINC150 Dataset")
    st.markdown("Evaluate the system's performance on Out-of-Distribution (OOD) detection and LLM labeling using the CLINC150 dataset.")
    
    # Evaluation Configuration
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        sample_size = st.number_input(
            "Sample Size for Evaluation",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Number of samples to use for evaluation (includes both in-domain and OOS samples)"
        )
    
    with col_eval2:
        include_oos = st.checkbox("Include Out-of-Scope (OOS) Samples", value=True,
                                help="Include OOS samples to test OOD detection performance")
    
    # Evaluation Buttons
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("Run Evaluation", type="primary"):
            if 'orchestrator' not in st.session_state:
                st.error("Please initialize the system first by bootstrapping a model.")
            else:
                with st.spinner("Running evaluation... This may take a few minutes."):
                    try:
                        # اجرای ارزیابی
                        evaluator = AutoMTLEvaluator(st.session_state.orchestrator)
                        metrics = evaluator.evaluate_system(sample_size=sample_size, include_oos=include_oos)
                        
                        # ذخیره نتایج در session state
                        st.session_state.evaluation_metrics = metrics
                        st.session_state.evaluation_results = evaluator.get_detailed_results()
                        
                        st.success("Evaluation completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error during evaluation: {e}")
                        import traceback
                        traceback.print_exc()
    
    with col_btn2:
        if st.button("Save Results"):
            if 'evaluation_metrics' in st.session_state:
                try:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"evaluation_results_{timestamp}.json"
                    
                    # ذخیره نتایج
                    import json
                    results_data = {
                        'metrics': st.session_state.evaluation_metrics,
                        'configuration': {
                            'sample_size': sample_size,
                            'include_oos': include_oos,
                            'timestamp': timestamp
                        }
                    }
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(results_data, f, ensure_ascii=False, indent=2)
                    
                    st.success(f"Results saved to {filename}")
                    
                except Exception as e:
                    st.error(f"Error saving results: {e}")
            else:
                st.warning("No evaluation results to save. Run evaluation first.")
    
    # Display Results
    if 'evaluation_metrics' in st.session_state:
        metrics = st.session_state.evaluation_metrics
        
        st.divider()
        st.subheader("📊 Evaluation Results")
        
        # Summary Statistics
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            st.metric("Total Samples", metrics['overall']['total_samples'])
        
        with col_summary2:
            st.metric("Fallback Rate", f"{metrics['overall']['fallback_rate']:.1%}")
        
        with col_summary3:
            st.metric("Evaluation Time", f"{metrics['evaluation_time']:.1f}s")
        
        # OOD Detection Performance
        st.subheader("🔍 OOD Detection Performance")
        ood_metrics = metrics['ood_detection']
        
        col_ood1, col_ood2, col_ood3, col_ood4 = st.columns(4)
        
        with col_ood1:
            st.metric("OOD Accuracy", f"{ood_metrics['accuracy']:.1%}")
        
        with col_ood2:
            st.metric("OOD Precision", f"{ood_metrics['precision']:.1%}")
        
        with col_ood3:
            st.metric("OOD Recall", f"{ood_metrics['recall']:.1%}")
        
        with col_ood4:
            st.metric("OOD F1-Score", f"{ood_metrics['f1_score']:.1%}")
        
        # LLM Labeling Performance
        if metrics['llm_labeling']['total_fallbacks'] > 0:
            st.subheader("🏷️ LLM Labeling Performance")
            llm_metrics = metrics['llm_labeling']
            
            col_llm1, col_llm2, col_llm3, col_llm4 = st.columns(4)
            
            with col_llm1:
                st.metric("LLM Accuracy", f"{llm_metrics['accuracy']:.1%}")
            
            with col_llm2:
                st.metric("LLM Precision", f"{llm_metrics['precision']:.1%}")
            
            with col_llm3:
                st.metric("LLM Recall", f"{llm_metrics['recall']:.1%}")
            
            with col_llm4:
                st.metric("LLM F1-Score", f"{llm_metrics['f1_score']:.1%}")
            
            st.metric("Total Fallbacks", llm_metrics['total_fallbacks'])
        
        # Detailed Results Table
        with st.expander("📋 Detailed Results"):
            if 'evaluation_results' in st.session_state:
                results_df = st.session_state.evaluation_results
                
                # نمایش نمونه‌های جالب
                st.write("Sample Predictions:")
                
                # نمونه‌هایی که به درستی OOD شناسایی شدند
                correct_ood = results_df[
                    (results_df['is_true_oos'] == True) & 
                    (results_df['is_predicted_oos'] == True)
                ].head(5)
                
                if not correct_ood.empty:
                    st.write("✅ Correctly detected OOD samples:")
                    st.dataframe(correct_ood[['text', 'true_label', 'predicted_label', 'confidence']])
                
                # نمونه‌هایی که اشتباه OOD شناسایی شدند
                false_ood = results_df[
                    (results_df['is_true_oos'] == False) & 
                    (results_df['is_predicted_oos'] == True)
                ].head(5)
                
                if not false_ood.empty:
                    st.write("❌ False OOD detections:")
                    st.dataframe(false_ood[['text', 'true_label', 'predicted_label', 'confidence']])
                
                # نمونه‌هایی که به LLM ارجاع شدند
                llm_fallbacks = results_df[results_df['is_fallback'] == True].head(5)
                
                if not llm_fallbacks.empty:
                    st.write("🤖 Samples that fell back to LLM:")
                    st.dataframe(llm_fallbacks[['text', 'true_label', 'predicted_label', 'llm_label_correct']])
    
    else:
        st.info("💡 Click 'Run Evaluation' to start the system evaluation process.")
