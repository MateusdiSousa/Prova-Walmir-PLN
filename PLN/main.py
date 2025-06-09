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
def criar_banco_vetorial(nome_colecao="colecao_teste"):
    # Gerar embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2') ## instancia o modelo de geração de embeddings
    
    # Criar banco vetorial
    client = chromadb.Client() ## instância o banco de dados
    collection = client.create_collection(name=nome_colecao) ## cria a collection
    
    return collection, model

def adicionar_chunks(chunks, collection, model):
    embeddings = model.encode(chunks) ## transforma os chunks em embeddings para serem armazenados no banco vetorial
    ids = [f"doc_{i}" for i in range(len(chunks))] ## Cria os ids dos documentos a partir da quantidade de chunks que serão armazenados
    
    # Adicionar documentos
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),  
        ids=ids
    ) ## adiciona os dados vetorizados na collection
    return

# 5. Função de consulta
def consultar_banco(colecao, modelo, consulta, n_resultados=1):
    embedding_consulta = modelo.encode([consulta]) ## transforma a consulta em um valor dado vetoriazado para realizar a busca no banco
    resultados = colecao.query(
        query_embeddings=embedding_consulta.tolist(),
        n_results=n_resultados ## quantidade de resultados que serão retornados
    )
    return resultados


def ExibirResultados(resultados):
    for i in range(len(resultados['ids'][0])):
        print(f"ID: {resultados['ids'][0][i]}")
        print(f"Documento: {resultados['documents'][0][i]}")
        print(f"Distância: {resultados['distances'][0][i]}")
        print("-" * 40)
# Fluxo principal
if __name__ == "__main__":
    print("\n============================== CSV =====================================\n")
    
    # 1º Passo Extrair texto do CSV
    texto_csv = combinar_colunas_csv("arquivos/people-100.csv", ["First Name", "Last Name", "Job Title"])

    # 2º Passo: Pré processar o texto
    texto_csv_tratado = tratamento_pln(texto=texto_csv)

    # 3º Passo: Transformar o texto em chunks
    chunks_csv = criar_chunks(texto_csv_tratado, tamanho=40, overlap=10)

    # 4º Passo: Criar o banco vetorial e o modelo de embedding
    colecao_people, modelo1 = criar_banco_vetorial(nome_colecao="peolple")

    # 5º Passo: Transformar os chunks em embeddings e adicionar ao banco vetorial
    adicionar_chunks(chunks=chunks_csv, collection=colecao_people, model=modelo1)

    # 6º Passo: Realizar query no banco de dados
    resultados_csv = consultar_banco(colecao=colecao_people, modelo=modelo1, consulta="Game Developer", n_resultados=3)
    ExibirResultados(resultados=resultados_csv)

    print("\n============================== PDF =====================================\n")

    # 1º Passo Extrair texto do PDF
    texto_porquinhos = ler_pdf("arquivos/os_3_porquinhos.pdf")

    # 2º Passo: Pré processar o texto
    texto_tratado_porquinhos = tratamento_pln(texto=texto_porquinhos)

    # 3º Passo: Transformar o texto em chunks
    chunks = criar_chunks(texto_tratado_porquinhos, tamanho=100, overlap=30)

    # 4º Passo: Criar o banco vetorial e o modelo de embedding
    colecao, modelo = criar_banco_vetorial(nome_colecao="tres_porquinhos")

    # 5º Passo: Transformar os chunks em embeddings e adicionar ao banco vetorial
    adicionar_chunks(chunks=chunks, collection=colecao, model=modelo)

    # 6º Passo: Realizar query no banco de dados
    resultados = consultar_banco(colecao=colecao, modelo=modelo, consulta="lobo mal assoprou", n_resultados=3)

    # # Exibir resultados
    ExibirResultados(resultados=resultados)    