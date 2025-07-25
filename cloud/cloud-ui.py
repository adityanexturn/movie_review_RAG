import os
import httpx
import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# --- Helper: Convert image to base64 for inline logo display ---
def logo_img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- Load environment variables ---
load_dotenv()

# Get secrets from Streamlit Cloud or environment variables
def get_secret(key):
    try:
        return st.secrets[key]  # For Streamlit Cloud deployment
    except:
        return os.getenv(key)   # For local development with .env file

api_key = get_secret("GROQ_API_KEY")
weaviate_url = get_secret("WEAVIATE_CLUSTER_URL")
weaviate_api_key = get_secret("WEAVIATE_API_KEY")

# --- Streamlit UI setup ---
st.set_page_config(page_title="FILM REVIEW RAG", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Quick Actions")
    if st.button("Exit", type="primary"):
        st.warning("Session ended. You may close this tab.")
        st.stop()

    if st.button("Clear Chat"):
        st.session_state.chat = []
        st.session_state.evidence = []
        st.success("Chat history cleared.")

    st.markdown("---")
    with st.expander("Sample Questions"):
        st.write("- What do reviewers say about The Prestige?")
        st.write("- Is Leonardo DiCaprio praised in Shutter Island?")
        st.write("- Who are the actors of Bareilly Ki Barfi?")
        st.write("- What are themes in Ayushmann Khurrana's films?")

# --- Header: Centered + animated logo ---
def display_header_with_logo():
    # Try multiple logo paths (local and cloud)
    logo_paths = [
        r"C:\Users\adity\Desktop\Gen-Ai Rag\code\critic.png",  # Local path
        "critic.png",  # Cloud path (in repository)
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
                        25% {{ transform: translateY(-8px) scale(1.05); }}
                        50% {{ transform: translateY(-12px) scale(1.1); }}
                        75% {{ transform: translateY(-8px) scale(1.05); }}
                        100% {{ transform: translateY(0px) scale(1); }}
                    }}
                    
                    @keyframes glow {{
                        0%, 100% {{ 
                            box-shadow: 0 0 5px rgba(255, 107, 107, 0.3),
                                       0 0 10px rgba(78, 205, 196, 0.3),
                                       0 0 15px rgba(255, 107, 107, 0.2);
                        }}
                        50% {{ 
                            box-shadow: 0 0 20px rgba(255, 107, 107, 0.6),
                                       0 0 30px rgba(78, 205, 196, 0.6),
                                       0 0 40px rgba(255, 107, 107, 0.4);
                        }}
                    }}
                    
                    .header-container {{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        gap: 20px;
                        padding: 2rem 0 3rem 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 15px;
                        margin-bottom: 2rem;
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
                    
                    @keyframes shimmer {{
                        0% {{ left: -100%; }}
                        100% {{ left: 100%; }}
                    }}
                    
                    .logo-img {{
                        width: 60px;
                        height: 60px;
                        animation: smoothBounce 2s ease-in-out infinite,
                                  glow 2s ease-in-out infinite alternate;
                        border-radius: 12px;
                        border: 2px solid rgba(255, 255, 255, 0.3);
                        transition: all 0.3s ease;
                        z-index: 2;
                    }}
                    
                    .logo-img:hover {{
                        transform: scale(1.2) rotate(5deg);
                        animation-play-state: paused;
                    }}
                    
                    .title-text {{
                        font-size: 3rem;
                        font-weight: bold;
                        margin: 0;
                        color: white;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                        z-index: 2;
                        letter-spacing: 2px;
                    }}
                    
                    .subtitle-text {{
                        font-size: 1rem;
                        color: rgba(255, 255, 255, 0.8);
                        margin-top: 0.5rem;
                        font-style: italic;
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

# Call the header function
logo_loaded = display_header_with_logo()

# # Fallback header if logo fails to load
# if not logo_loaded:
#     st.markdown(
#         """
#         <style>
#         .fallback-header {
#             text-align: center;
#             padding: 2rem 0 3rem 0;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             border-radius: 15px;
#             margin-bottom: 2rem;
#         }
#         .fallback-title {
#             font-size: 3rem;
#             font-weight: bold;
#             margin: 0;
#             color: white;
#             text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
#             letter-spacing: 2px;
#         }
#         .movie-emoji {
#             font-size: 4rem;
#             margin-bottom: 1rem;
#             animation: spin 3s linear infinite;
#         }
#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }
#         </style>
#         <div class="fallback-header">
#             <div class="movie-emoji">🎬</div>
#             <div class="fallback-title">Film Review RAG</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# --- Weaviate and embedding model ---
@st.cache_resource
def get_clients():
    try:
        # Connect to Weaviate Cloud
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
if clients[0] is None or clients[1] is None:
    st.stop()

client, embed_model = clients
chunk_collection = client.collections.get("Chunk")

# --- Few-shot examples ---
few_shot_examples = [
    {
        "question": "What do reviewers think of Inception?",
        "answer": "Many reviewers found Inception to be visually stunning, intellectually engaging, and emotionally powerful."
    },
    {
        "question": "How was Heath Ledger's performance in The Dark Knight?",
        "answer": "Reviewers widely praised Heath Ledger's performance as iconic, dark, and transformative, often calling it the highlight of the film."
    }
]

# --- Prompt builders ---
def build_prompt(evidence_chunks, user_question):
    prompt = (
        "You are a movie review assistant. You must answer ONLY using the evidence provided below.\n"
        "Extract and provide information directly from the evidence, even if it seems incomplete or incorrect.\n"
        "Always try to answer based on what's explicitly stated in the evidence.\n"
        "Only say some or the other answer even if the answer in the evidence is incorrect.\n\n"
    )
    for ex in few_shot_examples:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt += "Evidence:\n"
    for i, chunk in enumerate(evidence_chunks):
        prompt += f"{i+1}. {chunk.properties['text']} (Source: {chunk.properties['source']})\n"
    prompt += f"\nQ: {user_question}\nA:"
    return prompt

def is_relevant(chunk_text, important_terms):
    return any(term in chunk_text.lower() for term in important_terms)

# --- Session state ---
if "chat" not in st.session_state:
    st.session_state.chat = []
    st.session_state.evidence = []

# --- Main chat input ---
user_input = st.chat_input("Ask a question using the reviews...")

if user_input:
    basic_stopwords = {"the", "and", "for", "with", "this", "that", "from", "have", "has", "was", "are", "but"}
    important_terms = [
        word.lower() for word in user_input.lower().split()
        if len(word) > 3 and word not in basic_stopwords
    ]

    query_vector = embed_model.encode(user_input)

    with st.spinner("Finding relevant evidence and generating answer..."):
        try:
            results = chunk_collection.query.near_vector(
                near_vector=query_vector,
                limit=10,
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
            top_chunks = filtered_chunks[:4] if filtered_chunks else dedup_chunks[:1]
            st.session_state.evidence.append(top_chunks)

            prompt = build_prompt(top_chunks, user_input)
            system_message = (
                "You are a film review assistant. Answer using the provided evidence. "
                "Extract and state information directly from the evidence, even if it appears incomplete. "
                "Only say 'Not enough information' if the question topic is completely absent from the evidence. "
                "Do not use general knowledge beyond what's in the evidence."
            )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 512
            }

            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )

            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"].strip()
                st.session_state.chat.append(("user", user_input))
                st.session_state.chat.append(("assistant", answer))
            else:
                st.error(f"Error {response.status_code}: Unable to generate answer.")
        except Exception as e:
            st.error(f"Unexpected error occurred: {e}")

# --- Display chat history and evidence ---
if st.session_state.chat:
    for i, (speaker, msg) in enumerate(st.session_state.chat):
        if speaker == "user":
            st.markdown(
                f"""
                <div style='
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 12px 18px;
                    border-radius: 12px;
                    margin-bottom: 10px;
                    color: white;
                    max-width: 90%;
                    font-size: 16px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                '>
                <b>🎬 You:</b><br>{msg}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='
                    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    padding: 12px 18px;
                    border-radius: 12px;
                    margin-bottom: 10px;
                    color: white;
                    max-width: 90%;
                    font-size: 16px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                '>
                <b>🤖 Assistant:</b><br>{msg}
                </div>
                """,
                unsafe_allow_html=True
            )
            idx = i // 2
            if idx < len(st.session_state.evidence):
                st.markdown("**📚 Evidence Sources:**")
                for j, chunk in enumerate(st.session_state.evidence[idx]):
                    st.markdown(f"**📄 Evidence {j+1}:** {chunk.properties['source']}")
                    with st.expander("📖 View Chunk Text", expanded=False):
                        st.markdown(
                            f"""
                            <div style="
                                max-height: 300px;
                                overflow-y: auto;
                                overflow-x: hidden;
                                padding: 10px;
                                background-color: #87ceeb;
                                border-radius: 8px;
                                border-left: 4px solid #007acc;
                                font-family: 'Courier New', monospace;
                                font-size: 14px;
                                line-height: 1.5;
                                white-space: pre-wrap;
                                word-wrap: break-word;
                            ">
                            {chunk.properties['text']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

# # --- Footer Credit ---
# st.markdown(
#     """
#     <style>
#     .creator-credit {
#         position: fixed;
#         bottom: 20px;
#         right: 20px;
#         z-index: 9999;
#         font-size: 1rem;
#         color: white;
#         opacity: 0.9;
#         pointer-events: none;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 8px 16px;
#         border-radius: 25px;
#         backdrop-filter: blur(10px);
#         box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
#         font-weight: 600;
#         letter-spacing: 0.5px;
#         border: 2px solid rgba(255, 255, 255, 0.2);
#     }
#     .creator-credit::before {
#         content: '🎬 ';
#         margin-right: 5px;
#     }
#     </style>
#     <div class="creator-credit">Created by Aditya</div>
#     """,
#     unsafe_allow_html=True
# )
