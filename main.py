import streamlit as st
from rag_chain import ingest_documents, build_qa_chain
from evaluation import bleu_score, rouge_score_fn, token_based_precision_recall_f1

st.title("🔍 RAG Chatbot with Evaluation")
st.set_page_config(page_title="RAG Chatbot with Evaluation", layout="wide")

question = st.text_input("Ask a question:")

if question:
    vectorstore = ingest_documents()
    qa_chain = build_qa_chain(vectorstore)

    response = qa_chain.run(question)
    st.markdown(f"**💬 Answer:** {response}")

    ref_answer = st.text_input("✏️ Enter reference answer (for eval):")
    if ref_answer:
        st.subheader("📊 Evaluation Metrics")
        st.write(f"BLEU: {bleu_score(response, ref_answer):.2f}")
        rouge = rouge_score_fn(response, ref_answer)
        st.write(f"ROUGE-1: {rouge['rouge1'].fmeasure:.2f}, ROUGE-L: {rouge['rougeL'].fmeasure:.2f}")
        prf = token_based_precision_recall_f1(response, ref_answer)
        st.write(f"Precision: {prf['precision']:.2f}, Recall: {prf['recall']:.2f}, F1: {prf['f1']:.2f}")
