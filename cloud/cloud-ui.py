import os
import re
import httpx
import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import plotly.graph_objects as go


# --- Helper Functions ---
def logo_img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)


# --- Load Environment Variables ---
load_dotenv()
api_key = get_secret("GROQ_API_KEY")
weaviate_url = get_secret("WEAVIATE_CLUSTER_URL")
weaviate_api_key = get_secret("WEAVIATE_API_KEY")


# --- Streamlit UI Setup ---
st.set_page_config(page_title="FILM REVIEW RAG", layout="centered")


# --- Sidebar ---
with st.sidebar:
    
    if st.button("Exit Application", type="primary", use_container_width=True):
        st.warning("Session ended. You may close this tab.")
        st.stop()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat = []
        st.session_state.evidence = []
        st.success("Chat history cleared.")

    st.markdown("---")
    with st.expander("Sample Questions"):
        st.write("- How was Heath Ledger's performance in The Dark Knight?")
        st.write("- Is Leonardo DiCaprio praised in Shutter Island?")
        st.write("- How is the Romance in Me Before You?")
        st.write("- What are themes in Ayushmann Khurrana's films?")

    # --- EVALUATION SECTION ---
    st.markdown("---")
    st.markdown("### Manual Evaluation")
    with st.expander("CSV Evaluation", expanded=True):
        st.markdown("Upload CSV with `question`, `ground_truth`, `system_ans` columns")
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Standardize column names
                column_mapping = {}
                for col in df.columns:
                    if col.lower() == 'question':
                        column_mapping[col] = 'question'
                    elif col.lower() == 'ground_truth':
                        column_mapping[col] = 'ground_truth'
                    elif col.lower() in ['system_ans', 'system', 'system_answer']:
                        column_mapping[col] = 'system_ans'

                df = df.rename(columns=column_mapping)
                
                required_cols = ['question', 'ground_truth', 'system_ans']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {required_cols}")
                    st.stop()

                st.success(f"Loaded {len(df)} evaluation questions")

                # --- EVALUATION  ---
                def adjusted_evaluation_logic(ground_truth, system_answers):
                    correct_predictions = []
                    
                    for gt, sys in zip(ground_truth, system_answers):
                        gt_str = str(gt).strip()
                        sys_str = str(sys).strip()
                        
                        # Normalize text
                        gt_clean = re.sub(r'\s+', ' ', gt_str.lower().strip())
                        sys_clean = re.sub(r'\s+', ' ', sys_str.lower().strip())
                        
                        # 1. EXACT MATCH
                        if gt_clean == sys_clean:
                            correct_predictions.append(1)
                            continue
                        
                        # 2. HANDLE "NOT ENOUGH INFORMATION" CASES
                        not_enough_phrases = ['not enough information', 'cannot answer', 'no information', 'insufficient information', 'evidence does not state']
                        gt_has_not_enough = any(phrase in gt_clean for phrase in not_enough_phrases)
                        sys_has_not_enough = any(phrase in sys_clean for phrase in not_enough_phrases)
                        
                        if not gt_has_not_enough and sys_has_not_enough:
                            correct_predictions.append(0)
                            continue
                            
                        if gt_has_not_enough and sys_has_not_enough:
                            correct_predictions.append(1)
                            continue
                        
                        if gt_has_not_enough and not sys_has_not_enough:
                            correct_predictions.append(0)
                            continue
                        
                        # 3. CONTAINMENT CHECK 
                        if len(sys_clean) >= 3 and (sys_clean in gt_clean or gt_clean in sys_clean):
                            correct_predictions.append(1)
                            continue
                        
                        # 4. ENTITY MATCHING 
                        gt_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', gt_str))
                        sys_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sys_str))
                        
                        non_entities = {'The', 'Movie', 'Film', 'Actor', 'Director', 'Based', 'According', 'Evidence', 'Source', 'Review'}
                        gt_entities = gt_entities - non_entities
                        sys_entities = sys_entities - non_entities
                        
                        if gt_entities and sys_entities:
                            entity_overlap = len(gt_entities.intersection(sys_entities)) / len(gt_entities)
                            if entity_overlap >= 0.6:  
                                correct_predictions.append(1)
                                continue
                        
                        # 5. SEMANTIC CONTENT MATCHING (Lowered to 70%)
                        gt_words = set(re.findall(r'\b\w+\b', gt_clean))
                        sys_words = set(re.findall(r'\b\w+\b', sys_clean))
                        
                        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'has', 'have', 'had'}
                        gt_words = gt_words - stop_words
                        sys_words = sys_words - stop_words
                        
                        if not gt_words:
                            correct_predictions.append(0)
                            continue
                        overlap = len(gt_words.intersection(sys_words))
                        overlap_ratio = overlap / len(gt_words)
                        
                        if overlap_ratio >= 0.70:  
                            correct_predictions.append(1)
                        else:
                            correct_predictions.append(0)
                    
                    return correct_predictions

                # --- RUN EVALUATION ---
                if st.button("Run Evaluation"):
                    with st.spinner("Running evaluation..."):
                        correct_predictions = adjusted_evaluation_logic(df['ground_truth'], df['system_ans'])
                        
                        # Create binary classification for confusion matrix
                        y_true = [1] * len(df)  # All should be correct (ground truth)
                        y_pred = correct_predictions
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        cm = confusion_matrix(y_true, y_pred)
                        
                        # Display metrics only
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.3f}")
                        with col2:
                            st.metric("Precision", f"{precision:.3f}")
                        with col3:
                            st.metric("Recall", f"{recall:.3f}")
                        with col4:
                            st.metric("F1 Score", f"{f1:.3f}")
                        
                        # Confusion Matrix only
                        st.markdown("#### Confusion Matrix")
                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Predicted Incorrect', 'Predicted Correct'],
                            y=['Actually Incorrect', 'Actually Correct'],
                            colorscale='Blues',
                            text=cm,
                            texttemplate="%{text}",
                            textfont={"size": 16},
                            hoverongaps=False
                        ))
                        
                        fig.update_layout(
                            title='Confusion Matrix',
                            xaxis_title='Predicted',
                            yaxis_title='Actual',
                            width=400,
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing file: {e}")


# --- Header Display ---
def display_header_with_logo():
    logo_paths = [
        r"C:\Users\adity\Desktop\Gen-Ai Rag\code\critic.png",
        "critic.png",
    ]
    
    for logo_path in logo_paths:
        try:
            if os.path.exists(logo_path):
                logo_img = Image.open(logo_path)
                logo_base64 = logo_img_to_base64(logo_img)
                
                st.markdown(
                    f"""
                    <style>
                    @keyframes smoothBounce {{
                        0% {{ transform: translateY(0px) scale(1); }}
                        25% {{ transform: translateY(-5px) scale(1.02); }}
                        50% {{ transform: translateY(-8px) scale(1.05); }}
                        75% {{ transform: translateY(-5px) scale(1.02); }}
                        100% {{ transform: translateY(0px) scale(1); }}
                    }}
                    
                    @keyframes shimmer {{
                        0% {{ left: -100%; }}
                        100% {{ left: 100%; }}
                    }}
                    
                    .header-container {{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 15px;
                        padding: 1rem 0 1.5rem 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 10px;
                        margin-bottom: 1.5rem;
                        position: relative;
                        overflow: hidden;
                    }}
                    
                    .header-container::before {{
                        content: '';
                        position: absolute;
                        top: 0;
                        left: -100%;
                        width: 100%;
                        height: 100%;
                        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                        animation: shimmer 3s infinite;
                    }}
                    
                    .logo-img {{
                        width: 45px;
                        height: 45px;
                        animation: smoothBounce 2s ease-in-out infinite;
                        border-radius: 8px;
                        border: 2px solid rgba(255, 255, 255, 0.3);
                        transition: all 0.3s ease;
                        z-index: 2;
                    }}
                    
                    .title-text {{
                        font-size: 2.5rem;
                        font-weight: bold;
                        margin: 0;
                        color: white;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                        z-index: 2;
                        letter-spacing: 1.5px;
                    }}
                    
                    .subtitle-text {{
                        font-size: 0.9rem;
                        color: rgba(255, 255, 255, 0.8);
                        margin-top: 0.3rem;
                        z-index: 2;
                    }}
                    </style>
                    <div class="header-container">
                        <img src="data:image/png;base64,{logo_base64}" class="logo-img" alt="Film Critic Logo" />
                        <div>
                            <div class="title-text">FILM REVIEW RAG</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                return True
        except Exception as e:
            continue
    
    return False


display_header_with_logo()


# --- RAG System Setup ---
@st.cache_resource
def get_clients():
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
        )
        
        if not client.is_ready():
            st.error("Failed to connect to Weaviate Cloud. Please check your credentials.")
            return None, None
            
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return client, model
    except Exception as e:
        st.error(f"Error connecting to services: {e}")
        return None, None


# Initialize clients
clients = get_clients()
if not all(clients):
    st.stop()


client, embed_model = clients
chunk_collection = client.collections.get("Chunk")


# --- RAG Functions ---
def build_prompt(evidence_chunks, user_question):
    prompt = (
        "You are a movie review assistant. Answer ONLY using the evidence below.\n"
        "For questions about character relationships or connections, carefully examine "
        "all evidence pieces and synthesize the information logically.\n"
        "If multiple pieces of evidence relate to the question, combine them thoughtfully.\n\n"
    )
    
    few_shot_examples = [
        {"question": "Who plays the character Teddy Daniels?", "answer": "Leonardo DiCaprio plays the character Teddy Daniels.."},
        {"question": "How was Heath Ledger's performance in The Dark Knight?", "answer": "Reviewers widely praised Heath Ledger's performance as iconic, dark, and transformative, often calling it the highlight of the film."}
    ]
    
    for ex in few_shot_examples:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    
    prompt += "Evidence (examine all pieces carefully):\n"
    for i, chunk in enumerate(evidence_chunks):
        prompt += f"{i+1}. {chunk.properties['text']} (Source: {chunk.properties['source']})\n\n"
    
    prompt += f"Question: {user_question}\n"
    prompt += "Answer (synthesize from evidence above):"
    return prompt


def expand_query_terms(user_input):
    query_lower = user_input.lower()
    if 'riddler' in query_lower:
        user_input += " villain puzzle riddle Batman antagonist"
    elif 'batman' in query_lower and ('who' in query_lower or 'what' in query_lower):
        user_input += " character superhero vigilante"
    return user_input


def is_relevant(chunk_text, important_terms):
    return any(term in chunk_text.lower() for term in important_terms)


# --- Chat Interface ---
if "chat" not in st.session_state:
    st.session_state.chat = []
if "evidence" not in st.session_state:
    st.session_state.evidence = []


user_input = st.chat_input("Ask a question...")


if user_input:
    basic_stopwords = {"the", "and", "for", "with", "this", "that", "from", "have", "has", "was", "are", "but"}
    important_terms = [word.lower() for word in user_input.lower().split() if len(word) > 3 and word not in basic_stopwords]

    expanded_input = expand_query_terms(user_input)
    query_vector = embed_model.encode(expanded_input)

    with st.spinner("Finding relevant evidence and generating answer..."):
        try:
            results = chunk_collection.query.near_vector(
                near_vector=query_vector,
                limit=15,
                return_properties=["text", "source"]
            )
            
            seen = set()
            dedup_chunks = []
            for chunk in results.objects:
                text = chunk.properties["text"].strip()
                if text and text not in seen:
                    seen.add(text)
                    dedup_chunks.append(chunk)

            filtered_chunks = [chunk for chunk in dedup_chunks if is_relevant(chunk.properties["text"], important_terms)]
            top_chunks = filtered_chunks[:4] if filtered_chunks else dedup_chunks[:2]
            st.session_state.evidence.append(top_chunks)

            prompt = build_prompt(top_chunks, user_input)
            
            system_message = (
                "You are a movie review assistant. Answer using the provided evidence. "
                "If you find relevant information in the evidence, provide a direct answer. "
                "Only say 'Not enough information' if the topic is completely absent from the evidence. "
                "Be confident in your answers when the evidence supports them."
            )

            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
                "temperature": 0.001,
                "max_tokens": 512
            }

            response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30.0)

            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"].strip()
                st.session_state.chat.append(("user", user_input))
                st.session_state.chat.append(("assistant", answer))
            else:
                st.error(f"Error {response.status_code}: Unable to generate answer.")
        except Exception as e:
            st.error(f"Unexpected error occurred: {e}")


# --- Display Chat History ---
if st.session_state.chat:
    for i, (speaker, msg) in enumerate(st.session_state.chat):
        if speaker == "user":
            st.markdown(
                f"""
                <div style='
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 10px 15px; 
                    border-radius: 10px; 
                    margin-bottom: 8px; 
                    color: white; 
                    max-width: 90%; 
                    font-size: 14px; 
                    box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
                    position: relative;
                '>
                <div style='display: flex; align-items: center; gap: 6px;'>
                    <span style='font-size: 16px;'>🧑‍💻</span>
                    <b>You:</b>
                </div>
                <div style='margin-top: 6px;'>{msg}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='
                    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 10px 15px; 
                    border-radius: 10px; 
                    margin-bottom: 8px; 
                    color: white; 
                    max-width: 90%; 
                    font-size: 14px; 
                    box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
                    position: relative;
                '>
                <div style='display: flex; align-items: center; gap: 6px;'>
                    <span style='font-size: 16px;'>🤖</span>
                    <b>Assistant:</b>
                </div>
                <div style='margin-top: 6px;'>{msg}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            idx = i // 2
            if idx < len(st.session_state.evidence):
                st.markdown("---")
                st.markdown("**Evidence Sources:**")
                for j, chunk in enumerate(st.session_state.evidence[idx]):
                    with st.expander(f"**Evidence {j+1}:** {chunk.properties['source']}", expanded=False):
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #262730;
                                color: #ffffff;
                                padding: 12px;
                                border-radius: 6px;
                                border-left: 4px solid #00d4aa;
                                font-family: 'Courier New', monospace;
                                font-size: 12px;
                                line-height: 1.4;
                                max-height: 250px;
                                overflow-y: auto;
                                white-space: pre-wrap;
                                word-wrap: break-word;
                                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                            ">
                            {chunk.properties["text"]}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
