import io
import os
import time
import streamlit as st # type:ignore
from pypdf import PdfReader # type:ignore
from docx import Document as Dt 
from Embedding_file import Embeddings  

class RAG:
    def __init__(self):
        st.markdown("<h2 style='text-align: center;'>Expected RAG System</h2>", unsafe_allow_html=True)
        if 'file' not in st.session_state:
            st.session_state['file'] = None

    @st.cache_resource
    def load_model(_self, persist_directory):
        return Embeddings(persist_directory_filename=persist_directory)

    def uploader_(self, file, file_name):
        if file:
            if "file_name" not in st.session_state:
                st.session_state['file_name'] = None 
            st.write("**Embedding process in progress...**")
            st.session_state["file_name"] = file_name
            with st.spinner("📥 Processing..."):
                time.sleep(2)
                model_instance = self.load_model(persist_directory=st.session_state['file_name'])
                embeddings = model_instance.Embedding_file(file=file)
                
                # Store embeddings in session
                if "Embeddings" not in st.session_state:
                    st.session_state["Embeddings"] = {}
                st.session_state["Embeddings"][file_name] = embeddings
                

                st.success(f" Embedding completed for `{file_name}`!")
                return embeddings
            
    def PDF_extracter(self, file_path):
        
        """
    This function is used to Excrate the text from the pdf
    
    Agrs:
        Pdf_file_path (str): The path to the pdf file 

    Returns:
        Str: This extract text content from the pdf
        """

        Reader = PdfReader(file_path)
        text_content = ""
        for page in Reader.pages:
            text_content += page.extract_text()
        
        return text_content
    
    def Query(self, query):
        model_instance = self.load_model(persist_directory=st.session_state['file_name'])
        _, words = model_instance.run(query=query)

        st.markdown("### 🔍 Retrieved Results")
        

        # Streaming-like display
        output_placeholder = st.empty()
        streamed_text = ""
        for word in words:
            streamed_text += word + ""
            output_placeholder.markdown(streamed_text)
            time.sleep(0.02)
        return streamed_text

    def main(self):
        mode = st.radio("Select Mode:", ["Training", "Retrieval 🔍"])

        if mode == "Training":
            file = st.file_uploader("📄 Upload your text file:", type=['pdf', 'docx', 'html'])
            if file:
                # Ensure log directory exists
                os.makedirs("log", exist_ok=True)

                file_name = file.name.lower()
                file_bytes = file.read()
                text_content = ""

                
                if file_name.endswith('.html'):
                    text_content = file_bytes.decode("utf-8", errors="ignore")

                
                elif file_name.endswith('.docx'):
                    try:
                        doc = Dt(io.BytesIO(file_bytes))
                        text_content = "\n".join(p.text for p in doc.paragraphs)
                    except KeyError as e:
                        raise ValueError(
                            f"The DOCX file seems corrupted or incomplete ({e}). "
                            "Please open it in Word or LibreOffice and re-save it before uploading."
                        )

                
                elif file_name.endswith('.pdf'):
                    text_content = self.PDF_extracter(io.BytesIO(file_bytes))

                else:
                    st.error(" Unsupported file format")
                    return

                # ✅ Save processed content to log/
                log_path = os.path.join("log", file_name)
                with open(log_path, "w", encoding="utf-8") as out:
                    out.write(text_content)

                # ✅ Store in Streamlit session
                st.session_state['file'] = file
                st.session_state['content'] = text_content

                # ✅ Call your uploader method
                self.uploader_(file=text_content, file_name=file_name)

                # st.success(f"✅ {file_name} processed and saved to {log_path}")


        else:  # Retrieval Mode
            if st.session_state.get('file_name'):
                st.write("### Query and Retrieval")
                query = st.text_input("Enter your query:")
                if query:
                    query = query.strip()
                    if query == "?":
                        query = query.replace("?", "")
                    self.Query(query)
            else:
                st.warning("⚠️ Please upload and embed a file in Training mode first!")

####----------------------------------------------- FUNCTION CALL ---------------------------------------------------------------###

if __name__ == "__main__":
    app = RAG()
    app.main()
