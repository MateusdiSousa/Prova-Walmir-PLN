def criar_dataframe_simples(colunas, dados, indice=None):
    """
    Cria um DataFrame simples do Pandas
    
    Args:
        colunas (list): Lista de nomes de colunas
        dados (list of lists): Lista de listas com os dados
        indice (list, optional): Lista com os índices
        
    Returns:
        pd.DataFrame: DataFrame criado
    """
    import pandas as pd
    if len(colunas) != len(dados):
        raise ValueError("Número de colunas não corresponde ao número de listas de dados")
    
    dados_dict = {col: valores for col, valores in zip(colunas, dados)}
    return pd.DataFrame(dados_dict, index=indice)

def criar_series_simples(dados, indice=None, nome=None):
    """
    Cria uma Series simples do Pandas
    
    Args:
        dados (list): Valores da Series
        indice (list, optional): Índices da Series
        nome (str, optional): Nome da Series
        
    Returns:
        pd.Series: Series criada
    """
    import pandas as pd
    return pd.Series(dados, index=indice, name=nome)

def ler_csv(caminho, coluna_indice=None):
    """
    Lê um arquivo CSV e retorna um DataFrame
    
    Args:
        caminho (str): Caminho para o arquivo CSV
        coluna_indice (str, optional): Nome da coluna para usar como índice
        
    Returns:
        pd.DataFrame: DataFrame com os dados do CSV
    """
    import pandas as pd
    return pd.read_csv(caminho, index_col=coluna_indice)

def salvar_csv(dataframe, caminho):
    """
    Salva um DataFrame em um arquivo CSV
    
    Args:
        dataframe (pd.DataFrame): DataFrame a ser salvo
        caminho (str): Caminho para salvar o arquivo
    """
    dataframe.to_csv(caminho)
    print(f"DataFrame salvo com sucesso em {caminho}")
    
def selecionar_por_indice(dataframe, linhas=None, colunas=None):
    """
    Seleciona dados usando indexação numérica (iloc)
    
    Args:
        dataframe (pd.DataFrame): DataFrame de entrada
        linhas (int, slice or list, optional): Seleção de linhas
        colunas (int, slice or list, optional): Seleção de colunas
        
    Returns:
        pd.DataFrame or pd.Series: Dados selecionados
    """
    if linhas is None and colunas is None:
        return dataframe.iloc[:, :]
    elif colunas is None:
        return dataframe.iloc[linhas, :]
    elif linhas is None:
        return dataframe.iloc[:, colunas]
    else:
        return dataframe.iloc[linhas, colunas]

def selecionar_por_rotulo(dataframe, linhas=None, colunas=None):
    """
    Seleciona dados usando rótulos (loc)
    
    Args:
        dataframe (pd.DataFrame): DataFrame de entrada
        linhas (label, slice or list, optional): Seleção de linhas
        colunas (label, slice or list, optional): Seleção de colunas
        
    Returns:
        pd.DataFrame or pd.Series: Dados selecionados
    """
    if linhas is None and colunas is None:
        return dataframe.loc[:, :]
    elif colunas is None:
        return dataframe.loc[linhas, :]
    elif linhas is None:
        return dataframe.loc[:, colunas]
    else:
        return dataframe.loc[linhas, colunas]
    
def plotar_jointplot(dataframe, x, y, hue=None):
    """
    Cria um jointplot usando Seaborn
    
    Args:
        dataframe (pd.DataFrame): DataFrame com os dados
        x (str): Nome da coluna para o eixo x
        y (str): Nome da coluna para o eixo y
        hue (str, optional): Nome da coluna para agrupamento por cor
    """
    import seaborn as sns
    sns.jointplot(data=dataframe, x=x, y=y, hue=hue)
    import matplotlib.pyplot as plt
    plt.show()

def mostrar_primeiras_linhas(dataframe, n=5):
    """
    Mostra as primeiras n linhas de um DataFrame
    
    Args:
        dataframe (pd.DataFrame): DataFrame de entrada
        n (int, optional): Número de linhas a mostrar
    """
    return dataframe.head(n)