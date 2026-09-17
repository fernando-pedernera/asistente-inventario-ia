import os
import glob
from dotenv import load_dotenv

# Use proper document loader for PDFs (might need PyPDF2 or similar if pypdf is installed)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno (GEMINI_API_KEY)
load_dotenv()

# Rutas de datos
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "faiss_index")

def create_vector_store():
    print("Creando base de datos vectorial para los manuales...")
    
    # 1. Cargar PDFs
    pdf_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.pdf"))
    documents = []
    for pdf in pdf_files:
        print(f"Cargando {os.path.basename(pdf)}...")
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
        
    if not documents:
        print("No se encontraron PDFs en data/raw/")
        return None

    # 2. Dividir el texto en fragmentos (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Textos divididos en {len(chunks)} fragmentos.")

    # 3. Crear embeddings usando HuggingFace y guardar en FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 4. Guardar localmente
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector Store guardado exitosamente en {VECTOR_STORE_PATH}")
    return vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(question: str):
    # Cargar Vector Store si existe, sino crearlo
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = create_vector_store()
        if not vector_store:
            return "No se pudo cargar la base de conocimientos."

    retriever = vector_store.as_retriever()
    
    # 5. Configurar el LLM (Gemini) - Usar un modelo disponible
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)
    
    # Prompt Template
    template = (
        "Eres un asistente virtual experto en el inventario de la empresa. "
        "Usa los siguientes fragmentos de contexto recuperado de los manuales "
        "para responder a la pregunta del usuario. "
        "Si no sabes la respuesta, di que no lo sabes. Sé conciso y profesional.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    
    # Crear la cadena RAG con LCEL
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Consultar
    print(f"\nPregunta: {question}")
    response = rag_chain.invoke(question)
    print(f"Respuesta: {response}\n")
    return response

if __name__ == "__main__":
    # Test simple del RAG
    print("--- Test del Pipeline RAG ---")
    create_vector_store()
    get_answer("¿Qué precauciones debo tener al lavar la taza de viaje (travel)?")
