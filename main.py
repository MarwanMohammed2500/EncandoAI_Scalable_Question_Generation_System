from question_quality_control import sim_search
from embeddings import create_embeddings_model, store_vector
from loaders import read_pdf, read_dot_txt
from pipeline import create_template, get_response_from_llm, dump_to_dot_json
import streamlit as st
import asyncio
import tempfile
import random
import os

st.title("Questions Generator")    
# Upload files
uploaded_file = st.file_uploader(label="Upload a PDF or TXT file",
                                type=["pdf", "txt"],
                                accept_multiple_files=False
                                )

# Initialize embeddings model globally (only once)
@st.cache_resource
def init_embeddings_model():
    """
    Initialize the embeddings model once and load it in the cache to avoid loops
    """
    out_dim = 768
    embeddings_model_name = "models/gemini-embedding-001"
    try:
        # Create a new event loop for the async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return create_embeddings_model(out_dim=out_dim, model=embeddings_model_name)
    except Exception as e:
        st.error(f"Failed to initialize embeddings model: {e}")
        return None

def load_file(uploaded_file):
    """
    Get the file from the user (uploaded_file) and load that into the system from processing
    """
    if uploaded_file is None:
        return None, None
        
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        if file_extension == ".pdf":
            chunks = read_pdf(file_path=tmp_file_path)
        elif file_extension == ".txt":
            chunks = read_dot_txt(file_path=tmp_file_path)  # Fixed function name
        else:
            st.error("Unsupported file format")
            return None, None
        
        # Get embeddings model
        embeddings_model = init_embeddings_model()
        if embeddings_model:
            vector_store = store_vector(documents=chunks, embeddings_model=embeddings_model)
            return chunks, vector_store
        else:
            st.error("Embeddings model not initialized")
            return chunks, None
    
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass

def prompt_template():
    """
    Just an alias to "create_prompt_template" function to keep it clean
    """
    template = """You are a college professor creating an exam for students.
            Create {num_questions} MCQs for an upcoming exam with the following qualities:
            - Exactly one correct answer grounded in the source text.
            - Three plausible but incorrect "distractor" options.
            - Do not include “All of the above” or “None of the above.”
            - The question must be clear and answerable based only on the source text.
            - Don't directly say things like "According to the source text" or "In Page X it was mentioned..."

            Each question should follow this JSON schema:
            json("Question Number" (an integer): json(
            "Question": "string",
            "Options": ["(a) Option 1", "(b) Option 2", "(c) Option 3", "(d) Option 4"],
            "Answer": "string"
            ))
            Respond ONLY in valid JSON format
            ---
            Source Text:
            {context}"""
    input_variables = ["num_questions", "context"]
    prompt_template = create_template(input_variables, template)
    return prompt_template

def call_llm(prompt_template, documents):
    """
    Prepares the arguments needed to call "get_response_from_llm"
    """
    random.seed(42)
    content = [document.page_content for document in documents]
    k = min(5, len(content))
    context = "\n\n".join(random.sample(content, k=k))
    
    llm_model = "gemini-2.5-flash"
    input_vars_dict = {"num_questions":2, "context":context}
    json_response = get_response_from_llm(prompt_template=prompt_template, model=llm_model, input_vars_dict=input_vars_dict)
    return json_response

def main():
    if st.button("Generate Questions"):
        # Read File
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                chunks, vector_store = load_file(uploaded_file)
            if chunks and vector_store:
                st.success("File processed successfully!")
                st.write(f"Extracted {len(chunks)} text chunks")
            
            # Call the model
            with st.spinner("Generating questions..."):
                    try:
                        template = prompt_template()
                        json_response = call_llm(template, chunks)
                        
                        # Display JSON Response
                        if json_response:
                            st.subheader("Generated Questions:")
                            for question_num, question_data in json_response.items():
                                with st.container():
                                    st.markdown("---")
                                    st.markdown(f"### Question {question_num}")
                                    
                                    # Question field
                                    st.markdown("**Question:**")
                                    st.info(question_data.get("Question", "No question text"))
                                    
                                    # Options field
                                    st.markdown("**Options:**")
                                    for option in question_data.get("Options", []):
                                        st.write(f"• {option}")
                                    
                                    # Answer field
                                    st.markdown("**Correct Answer:**")
                                    st.success(question_data.get("Answer", "No answer provided"))
                                    
                                    # Add some spacing
                                    st.markdown("<br>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating questions: {str(e)}")
                        st.exception(e)
if __name__ == "__main__":
    main()