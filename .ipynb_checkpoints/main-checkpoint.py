# Importações necessárias
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pdfplumber
import re
import spacy
from nltk.corpus import stopwords
import chromadb
import nltk
import csv
import pandas as pd
 

# nltk.download('stopwords')


# 1. Função para ler PDF
def ler_pdf(caminho_pdf):
    with pdfplumber.open(caminho_pdf) as leitor_pdf:
        texto = "".join([pagina.extract_text() for pagina in leitor_pdf.pages])
        leitor_pdf.close()
    return texto.replace("\n", " ")

# 1.1. Função para ler CSV
def ler_csv(caminho_csv):
    try:
        df = pd.read_csv(caminho_csv)
        return df
    except Exception as e:
        print(f"Erro  ao ler CSV: {e}")
        return ""

def processar_linhas_csv(caminho_csv, coluna_texto):
    df = pd.read_csv(caminho_csv)
    textos_tratados = []
    
    for texto in df[coluna_texto]:
        texto_processado = tratamento_pln(str(texto))
        textos_tratados.append(texto_processado)
    
    return textos_tratados

def combinar_colunas_csv(caminho_csv, colunas):
    df = pd.read_csv(caminho_csv).to_dict("records") ## transforma uma lista de dicionários
    texto_combinado = ""

    for obj in df:
        for coluna in colunas:
            texto_combinado += f"{obj[coluna]} " ## faz uma linha com os valores das colunas selecionadas
        texto_combinado += "\n" 
    return texto_combinado

def transformar_dataframe_lista(df : pd.DataFrame, coluna : str) :
    return df[coluna].astype(str).tolist()
 
# 2. Função de pré-processamento de texto
def tratamento_pln(texto):
    # Carregar modelo e stopwords
    nlp = spacy.load("pt_core_news_sm")
    stop_words = set(stopwords.words('portuguese'))
    
    # Normalização
    texto = texto.lower() ## deixa todas as letras minúsculas 
    texto = re.sub(r'[^a-zA-Záéíóú\s]', '', texto) # Remoção de números, pontuações e caracteres especiais, utilizando regex 
    
    # Tokenização e limpeza
    doc = nlp(texto) # tokenização do texto
    clean_tokens = [token.lemma_ for token in doc 
                   if token.text not in stop_words and not token.is_punct] # tokens lematizados e sem stop words e pontuações
    
    return ' '.join(clean_tokens)

# 3. Divisão em chunks
def criar_chunks(texto, tamanho=30, overlap=10):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho,
        chunk_overlap=overlap
    ) # instancia o modelo setando o tamanho dos chunks em 40 com o overlap de 10
    return splitter.split_text(texto) # retorna os chunks 

# 4. Geração de embeddings e armazenamento
def criar_banco_vetorial(chunks, nome_colecao="colecao_teste"):
    # Gerar embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2') ## instancia o modelo de geração de embeddings
    embeddings = model.encode(chunks) ## transforma os chunks em embeddings para serem armazenados no banco vetorial
    
    # Criar banco vetorial
    client = chromadb.Client() ## instância o banco de dados
    collection = client.create_collection(name=nome_colecao) ## cria a collection
    
    # Adicionar documentos
    ids = [f"doc_{i}" for i in range(len(chunks))] ## Cria os ids dos documentos a partir da quantidade de chunks que serão armazenados
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),  
        ids=ids
    ) ## adiciona os dados vetorizados na collection
    
    return collection, model

# 5. Função de consulta
def consultar_banco(colecao, modelo, consulta, n_resultados=1):
    embedding_consulta = modelo.encode([consulta]) ## transforma a consulta em um valor dado vetoriazado para realizar a busca no banco
    resultados = colecao.query(
        query_embeddings=embedding_consulta.tolist(),
        n_results=n_resultados ## quantidade de resultados que serão retornados
    )
    return resultados

# Fluxo principal
if __name__ == "__main__":
    # Passo 1: Extrair texto
    
    ## CSV
    texto_csv = combinar_colunas_csv("people-100.csv", ["First Name", "Last Name", "Job Title", "Sex"]) # extrai o dados das colunas do CSV em formato de string

    # print(texto_csv)

    ## PDF
    # texto = ler_pdf("chapeuzinho.pdf")
    
    # Passo 2: Pré-processar
    texto_tratado = tratamento_pln(texto_csv)
    
    # # Passo 3: Criar chunks
    chunks = criar_chunks(texto_tratado, 50)
    
    # # Passo 4: Banco vetorial
    colecao, modelo = criar_banco_vetorial(chunks, "peoples")
    
    # # Passo 5: Consulta de exemplo
    resultados = consultar_banco(colecao, modelo, "Audiological",10) 
    
    # # Exibir resultados
    for i in range(len(resultados['ids'][0])):
        print(f"ID: {resultados['ids'][0][i]}")
        print(f"Documento: {resultados['documents'][0][i]}")
        print(f"Distância: {resultados['distances'][0][i]}")
        print("-" * 40)