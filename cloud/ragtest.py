import os
import json
import httpx
import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

# === 1. LOAD ENV AND API KEY ===
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env file")

# === 2. CONNECT TO WEAVIATE ===
client = weaviate.connect_to_local()
if not client.is_ready():
    raise RuntimeError("Weaviate is not running. Start Docker.")

chunk_collection = client.collections.get("Chunk")

# === 3. LOAD EMBEDDING MODEL ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# === 4. FEW-SHOT EXAMPLES ===
few_shot_examples = [
    {
        "question": "What do reviewers think of Inception?",
        "answer": "Many reviewers consider Inception a visually striking and thought-provoking film with a deep plot."
    },
    {
        "question": "How is the acting in The Dark Knight?",
        "answer": "The acting in The Dark Knight is especially praised, notably Heath Ledger's iconic role as the Joker."
    }
]

# === 5. BUILD PROMPT FUNCTION ===
def build_prompt(evidence_chunks, user_question):
    prompt = "Use the following film review snippets to answer the question accurately.\n\n"
    for ex in few_shot_examples:
        prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    prompt += "Evidence:\n"
    for i, chunk in enumerate(evidence_chunks):
        prompt += f"{i+1}. {chunk.properties['text']} (Source: {chunk.properties['source']})\n"
    prompt += f"\nQ: {user_question}\nA:"
    return prompt

# print("\nWelcome to your Film Review Retrieval-Augmented Generation (RAG) assistant!")
# print("You can ask any question about the reviews, or type 'exit' to quit.")

try:
    while True:
        user_question = input("\n🎬 Your Question (or type 'exit' to quit): ")
        if user_question.strip().lower() == 'exit':
            print("\nThank you for using the RAG assistant! Goodbye!")
            break

        # === 7. ENCODE THE QUESTION ===
        query_vector = embed_model.encode(user_question)

        # === 8. QUERY WEAVIATE FOR RELEVANT CHUNKS (fetch more to dedupe) ===
        results = chunk_collection.query.near_vector(
            near_vector=query_vector,
            limit=10,  # Extra results for deduplication
            return_properties=["text", "source"]
        )

        # === 8A. REMOVE DUPLICATES ===
        seen_texts = set()
        unique_chunks = []
        for chunk in results.objects:
            text = chunk.properties['text'].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_chunks.append(chunk)
        retrieved_chunks = unique_chunks[:3]  # Top 3 unique evidence chunks

        # === 9. SHOW RETRIEVED CHUNKS ===
        print("\nTop Retrieved Evidence:\n")
        for chunk in retrieved_chunks:
            print(f"> {chunk.properties['text']} (Source: {chunk.properties['source']})\n")

        # === 10. BUILD RAG PROMPT ===
        final_prompt = build_prompt(retrieved_chunks, user_question)

        # === 11. SEND TO GROQ LLM ===
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant answering questions using film reviews."},
                {"role": "user", "content": final_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }

        print("\n🤖 Sending prompt to GroqCloud API...\n")
        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            print("Answer from Groq Model:\n")
            print(answer.strip())
        else:
            print(f"Error {response.status_code}: {response.text}")

        # === 14. ASK USER IF THEY WISH TO CONTINUE OR EXIT ===
        print("\nWould you like to ask another question or exit?")
        next_step = input("Type '1' to ask another question, '2' to exit: ")
        if next_step.strip() == '2':
            print("\nThank you for using the RAG assistant! Goodbye!")
            break

finally:
    client.close()
