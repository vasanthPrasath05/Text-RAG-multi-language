import os
import streamlit as st # type:ignore
import shutil
from google import genai
from pypdf import PdfReader # type:ignore
from Prompt import RAG_prompt
from langchain_community.vectorstores import Chroma  
from langchain_community.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter

API_KEY = os.getenv("GEMINI_API_KEY","YOUR_API_KEY_HERE")
class Embeddings:

    def __init__(self, persist_directory_filename,
                 Embedding_model= HuggingFaceEmbeddings(
                    model_name="nomic-ai/nomic-embed-text-v2-moe", model_kwargs={"trust_remote_code": True}
                    )
                ):
        
        self.persist_store = f"Vector/{persist_directory_filename}./chroma_db"
        os.makedirs(self.persist_store, exist_ok=True)
        self.model = Embedding_model
    
    def text_split_function(self, text_content):
        Splitter = RecursiveCharacterTextSplitter(separators='.',chunk_size = 1000, chunk_overlap= 100)
        text = Splitter.split_text(text_content)
        return text
    
    def pdf_extracter(self, file_path):
        """
    Extracts all text from a given PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content from the PDF.
    """
        Reader = PdfReader(file_path)
        text_content = ""
        for page in Reader.pages:
            text_content += page.extract_text()
            print (text_content)
        return text_content

    def Embedding_file(self, file):

        """
    This fuction is used to Embedd the text content and store on the vector db

    Args:
        Str: File as text content of the document

    Retrun:
        This fuction did not retruns any things, it just save the embedded vectors into the vectorDB
        """
        
        text_content = file 
        Split_txt = self.text_split_function(text_content)

        document = [
            Document(page_content=chunk, metadata={"source": str(i)})
            for i, chunk in enumerate(Split_txt)
        ]
        texts = [doc.page_content for doc in document]
        shutil.rmtree(self.persist_store, ignore_errors=True) # Clear existing directory before creating a new one
        Chroma.from_texts(
            texts=texts,
            persist_directory=self.persist_store,
            embedding=self.model,
            collection_metadata={"hnsw:space": "cosine"},
        )

        print("The Embedding got sucessfully saved")
        return True
        
    def Load_Query_(self, query, top_k = 3):
        """
    This function is used to collect the User query and get the related chunks from the vectorDb,
    for this it will use the cosine similarity score.

    Args:
        str: User give query to retrive the chunks
        int: Number of K, it used collect the number of top chunks.
    
    Return:
        This function will return the top_chunk which are most simillar one.
        """
        docsearch = Chroma(
            persist_directory=self.persist_store,
            embedding_function=self.model
        )
        retrive_docu = docsearch.similarity_search_with_score(query,k= top_k)
        filtered_docs = []
        seen_texts = set()

        for doc, score in retrive_docu:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen_texts:
                filtered_docs.append((doc,score))
                seen_texts.add(source)

        filtered_docs.sort(key=lambda x: x[1],reverse=True)
        print("\n**Retrieved Documents with Similarity Scores:**")
        for doc, score in filtered_docs:
            st.write(f"📖**Source**: {doc.metadata.get('source', 'Unknown Source')}, **score**:{score :.4f}")
            st.write(f"📄 **Content Preview**: {doc.page_content[:200]}...\n")
        return [doc for doc, _ in filtered_docs]
    

    def generation_function(self, query, context):

        """
    This fuction is used to generate the Answer from the retireved context

    Args:
        Context (str): The top retireved context the cosine similarity function
        Query (str): The user give query, to Arrange the the context in the proper Structured answer.
    
    Return:
        Str: It return the final out from the LLM part.
        """

        if isinstance(context, list):
            context = "\n\n".join([doc.page_content for doc in context])
        else:
            str(context)

        prompt_ = RAG_prompt(query=query, context=context)

        prompt_content= prompt_.System_prompt + prompt_.user_prompt
        
        if not prompt_content:
            raise ValueError(f"The promt file empty")
        print (f"the prompt: {prompt_content}")
        client = genai.Client(api_key=API_KEY)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_content
        )
        
        return response.text.strip()
    
    def run(self, query):
        top_simillary = self.Load_Query_(query=query)
        final_output = self.generation_function(context=top_simillary, query=query)
        
        return top_simillary, final_output
    
    def stream_words(final_text):
            for word in final_text.split():
                yield word + ""

## ----------------------------------------------- FUNCTION CALL --------------------------------------------------------------- ###

if __name__ == "__main__":
    eb = Embeddings(persist_directory_filename="FILE LOCATION FOR STORE THE VECTOR DB")
    file = "PLACE YOUR TEXT CONTENT HEREQ"
    query = "PLACE YOUR QUERY HERE"

    eb.Embedding_file(file=file)
    top, final, = eb.run(query=query)

    print("\n--- Top Retrieved Chunks ---")
    for doc in top:
        print(doc.page_content[:200], "\n")
    print("\n--- Final Answer ---")
    print(final)

