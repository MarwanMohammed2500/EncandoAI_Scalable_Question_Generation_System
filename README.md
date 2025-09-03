# MCQ Generator (MVP)

This project is a **minimum viable product (MVP)** for generating multiple-choice questions (MCQs) from large text documents such as lecture notes or reports. It combines document loading, embeddings, and large language model prompting into an end-to-end pipeline with a simple Streamlit interface.

The system demonstrates:

* **File ingestion**: PDF and TXT loaders with chunking for long documents.
* **Embeddings & vector store**: Google Gemini embeddings with in-memory vector storage (extensible to FAISS/Pinecone).
* **Question generation**: LLM-based generation with enforced JSON schema (question, options, correct answer).
* **Prototype UI**: Streamlit app for uploading documents and instantly generating questions.

This is not a production-ready platform, but a functional demo that shows how such a system could be built and scaled. Future improvements (e.g., retrieval-augmented generation, question quality control, persistent storage, and deployment pipelines) are outlined below.

---
## Future Improvements & Scalability Opportunities

This MVP demonstrates the end-to-end flow of generating MCQs from long documents. Given additional time and resources, the following improvements could make the system production-ready and scalable:

* **Efficient Document Handling**

  * Replace random sampling with semantic retrieval (vector similarity search).
  * Support chunking and sliding windows for very large files (100+ pages).
  * Cache embeddings to avoid repeated computation.

* **Quality Control**

  * Integrate semantic similarity scoring between questions and source text.
  * Add a lightweight classifier to filter low-quality or ambiguous questions.
  * Provide feedback loops (human-in-the-loop review for QA).

* **Scalability & Performance**

  * Transition from in-memory vector stores to persistent solutions (e.g., FAISS, Pinecone, Postgres + pgvector).
  * Parallelize generation for batch question creation.
  * Use asynchronous processing for faster request handling.

* **Deployment & Reliability**

  * Package as a containerized service (Docker + orchestration with Kubernetes).
  * Add experiment tracking (TensorBoard/MLflow) for monitoring quality improvements.
  * Implement basic logging and error handling for LLM output reliability.

* **Extended Features**

  * Option to control difficulty level (easy/medium/hard).
  * Export results to CSV/JSON for downstream integration (e.g., LMS platforms).
  * UI extensions: question editing, quiz assembly, user feedback.