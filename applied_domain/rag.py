import os
import numpy as np
import pandas as pd
import torch
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from transformers import pipeline

print(find_dotenv())
load_dotenv(find_dotenv(), override=True)

MEDQUAD_CSV    = r"C:\Personal\comp_proj\spider_ml_task_2\applied_domain\medquad_csv\medquad.csv"
GUIDELINES_DIR = r"C:\Personal\comp_proj\spider_ml_task_2\applied_domain\guidelines_db"   # WHO, CDC, NICE PDFs go here
FAISS_DIR      = r"C:\Personal\comp_proj\spider_ml_task_2\applied_domain\faiss_db"
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"

device = "cuda" if torch.cuda.is_available() else "cpu"

EMERGENCY_KEYWORDS = ["suicide", "suicidal", "overdose", "chest pain", "heart attack", "stroke","can't breathe", "cannot breathe", "kill myself", "dying", "unconscious", "want to die", "wish to die"]
UNSAFE_KEYWORDS = ["prescribe", "diagnose me", "what dose", "how much medication","can i take", "should i take", "diagnosis"]

def is_emergency(query):
    q = query.lower()
    return any(k in q for k in EMERGENCY_KEYWORDS)

def is_unsafe(query):
    q = query.lower()
    return any(k in q for k in UNSAFE_KEYWORDS)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

def load_medquad():
    csv = pd.read_csv(MEDQUAD_CSV)
    docs = []
    for i, row in csv.iterrows():
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        metadata = {
            "paper": str(row.get("source", "MedQuad")),
            "number": str(i+1),
            "focus_area": str(row.get("focus_area", "")),
            "type": "MedQuAD"
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def load_guidelines():
    if not os.path.exists(GUIDELINES_DIR):
        return []
    loader = DirectoryLoader(GUIDELINES_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    files = loader.load()
    for file in files:
        name = os.path.basename(file.metadata.get("source", ""))
        file.metadata["paper"] = name
        file.metadata["type"] = "Guideline"
    return files


def ingest_all():
    with st.spinner("Loading MedQuAD..."):
        docs = load_medquad()
        st.write(f"Loaded MedQuad.")

    with st.spinner("Loading medical guidelines..."):
        files = load_guidelines()
        papers = set()
        for file in files:
            name = os.path.basename(file.metadata.get("source", ""))
            file.metadata["paper"] = name
            papers.add(name)
        st.write(f"Loaded {len(papers)} guideline documents, {len(files)} pages.")

    all_docs = docs+ files
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=70)
    chunks= splitter.split_documents(all_docs)
    st.write(f"Total chunks: {len(chunks)}")
    return chunks


def build_vectorstore():
    chunks = ingest_all()
    if not chunks:
        return None
    with st.spinner("Building vectorstore..."):
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_DIR)
    st.write("Vectorstore ready.")
    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)


def retrieve_with_scores(vectorstore, query):
    return vectorstore.similarity_search_with_score(query, k=6)


def rerank(query, docs_with_scores):
    reranker = get_reranker()
    docs = [d for d, _ in docs_with_scores]
    pairs = [(query, d.page_content) for d in docs]
    rerank_scores = reranker.predict(pairs)

    ranked = sorted(zip(rerank_scores.tolist(), docs), key=lambda x: x[0],reverse=True)
    return [(rerank_scores, doc) for rerank_scores, doc in ranked[:3]]


def estimate_confidence(reranked_docs):
    scores = [s for s,_ in reranked_docs]
    avg_score =np.mean(scores)
    # CrossEncoder ms-marco scores roughly range -10 to 10 - a more intuitive approach
    confidence= max(0, min(100, (avg_score + 10) / 20 * 100))
    return round(confidence, 1)

def evaluate_retrieval(reranked_docs):
    scores = [s for s, _ in reranked_docs]
    top_score = scores[0]

    retrieval_success = top_score > 0
    
    # what fraction of retrieved chunks are relevant
    count = 0
    for s in scores:
        if s > 0:
            count += 1
    precision_at_k = count / len(scores)

    # mean reciprocal rank
    for i, (score, _) in enumerate(reranked_docs):
        if score > 0:
            mrr= round(1 /(i + 1), 3)
            break
    
    return {
        "top_score": round(float(top_score), 2),
        "retrieval_success": retrieval_success,
        "precision_at_k": round(precision_at_k, 2),
        "mean_reciprocal_rank": mrr
    }

@st.cache_resource
def get_nli_model():
    return pipeline(
        "text-classification",
        model="cross-encoder/nli-MiniLM2-L6-H768",
        top_k=None
    )

def evaluate_hallucination(answer, reranked_docs):
    nli = get_nli_model()
    entailment_scores = []

    for _, doc in reranked_docs:
        context = doc.page_content[:500]
        answer_trunc= answer[:300]

        result =nli(f"{context} [SEP] {answer_trunc}", truncation=True) #3 nested lists as reranked docs are 3
        print(result)
        result=result[0]
        for item in result:
            if item["label"] == "entailment":
                entailment_scores.append(item["score"])
                break

    if not entailment_scores:
        return {"label": "neutral", "max_entailment": 0.0, "avg_entailment": 0.0}

    max_score = max(entailment_scores)
    avg_score = np.mean(entailment_scores)

    label = "entailment" if max_score > 0.7 else "neutral" if max_score > 0.4 else "contradiction"

    return {
        "label": label,
        "max_entailment": round(max_score * 100, 1),
        "avg_entailment": round(avg_score * 100, 1)
    }

def generate_answer(query, reranked_docs, confidence):
    with st.spinner("Generating answer..."):
        context_parts = []
        for _ ,doc in reranked_docs:
            source = doc.metadata.get("paper", "unknown")
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        if confidence< 40:
            confidence_note = "The retrieved evidence has low relevance. Express uncertainty clearly and recommend consulting a professional."
        elif confidence < 70:
            confidence_note = "The retrieved evidence has moderate relevance. Be appropriately cautious in your response and recommend consulting a professional if required."
        else:
            confidence_note = ""

        prompt = f"""You are a trustworthy healthcare information assistant providing evidence-based medical information.

    Rules:
    - Answer only using the provided context. Do not make up information.
    - Cite sources inline properly, e.g. "According to MedQuAD..." or "WHO guidelines state..."
    - Clearly separate established facts from assumptions and mention it in the answer.
    - Never recommend specific medications, dosages, or make diagnoses.
    - Always recommend consulting a qualified healthcare professional for personal medical decisions.
    - If the context lacks sufficient information, say so explicitly.
    {confidence_note}

    Context:
    {context}

    Question: {query}

    Answer:"""

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=2000)
        )
        return response.text.strip()


def run():
    st.title("Healthcare Information Assistant")
    st.write("A chatbot retrieving information from MedQuAD and trusted medical guidelines to give grounded and reasoned answers on medical queries.")

    os.makedirs(FAISS_DIR, exist_ok=True)

    if "vectorstore" not in st.session_state:
        try:
            st.session_state.vectorstore = load_vectorstore()
            st.success("Vectorstore loaded.")
        except Exception:
            st.session_state.vectorstore = build_vectorstore()
            if st.session_state.vectorstore is None:
                st.error("Failed to build vectorstore.")
                return

    query = st.text_input("Ask a health question:")

    if query:
        if is_emergency(query):
            st.error("This appears to be a medical emergency. Please call medical services or go to your nearest emergency room immediately.")
            return

        if is_unsafe(query):
            st.warning("This assistant cannot provide prescriptions, dosage recommendations, or personal diagnoses. Please consult a qualified healthcare professional for the same.")
            return

        docs_with_scores = retrieve_with_scores(st.session_state.vectorstore, query)
        reranked_docs = rerank(query, docs_with_scores)
        confidence = estimate_confidence(reranked_docs)
        answer = generate_answer(query, reranked_docs, confidence)

        st.divider()
        st.subheader("Answer")
        st.write(answer)

        st.divider()
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Confidence", f"{confidence}%")
        with col2:
            if confidence >= 70:
                st.success("High confidence — strong evidence retrieved.")
            elif confidence>= 40 and confidence<=70:
                st.warning("Moderate confidence — consider verifying with a medical professional.")
            else:
                st.error("Low confidence — limited evidence found. Please consult a healthcare professional.")

        st.divider()

        retrieval_eval = evaluate_retrieval(reranked_docs)
        hallucination_eval = evaluate_hallucination(answer, reranked_docs)

        with st.expander("Show evaluation metrics"):
            st.subheader("Retrieval Evaluation")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Top Chunk Score", retrieval_eval["top_score"])
            with col2:
                st.metric("Retrieval Success", "Yes" if retrieval_eval["retrieval_success"] else "No")
            with col3:
                st.metric("Precision@K", f"{retrieval_eval['precision_at_k'] * 100}%")
            with col4:
                st.metric("Mean Reciprocal Rank", f"{retrieval_eval['mean_reciprocal_rank']}")

            st.subheader("Hallucination Evaluation")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Entailment", f"{hallucination_eval['max_entailment']}%")
            with col2:
                st.metric("Avg Entailment", f"{hallucination_eval['avg_entailment']}%")
            with col3:
                label = hallucination_eval["label"]
                if label == "entailment":
                    st.success(f"Hallucination Risk: Low")
                elif label== "neutral":
                    st.warning(f"Hallucination Risk: Moderate")
                else:
                    st.error(f"Hallucination Risk: High")
        st.caption("Always consult a qualified healthcare professional before making any medical decisions.")

        with st.expander("Show retrieved evidence"):
            for i, (_, doc) in enumerate(reranked_docs):
                source = doc.metadata.get("paper", "")
                doc_type = doc.metadata.get("type", "")
                number= doc.metadata.get("number", "")
                page = doc.metadata.get("page", "")
                focus = doc.metadata.get("focus_area", "")

                label = f"{i+1}. {source}"
                if page:
                    label += f" :Page {page}"
                if number:
                    label+= f" :Question {number}"
                if doc_type:
                    label+= f" | [{doc_type}]"
                if focus:
                    label += f" | {focus}"

                excerpt = doc.page_content[:400].replace("\n", " ").strip()
                st.subheader(f"{label}")
                st.write(f"{excerpt}...")
                st.divider()


if __name__ == "__main__":
    run()