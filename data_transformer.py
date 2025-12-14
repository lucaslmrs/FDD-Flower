import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURAÇÕES MANUAIS - ALTERE AQUI
# =============================================================================

# Nome do arquivo CSV a ser processado (deve estar na pasta 'data')
CSV_FILE_NAME = "DataLogger_1.csv"  # TODO: Altere para o nome do arquivo desejado

# Identificador do cliente/cenário
CLIENT_ID = 1  # TODO: Altere para o ID do cliente correspondente

# Flag indicando se este cenário contém falhas
HAS_FAULT = False  # TODO: Altere para True se o cenário contém falhas

# =============================================================================


def load_tag_config(config_path):
    """
    Carrega o arquivo de configuração de tags e cria um mapeamento
    PrimaryKey -> Nome da Variável
    """
    tag_mapping = {}
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Pula a primeira linha (00000011)
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 7:
            primary_key = int(parts[0])
            # O último elemento é o caminho completo da variável
            full_path = parts[6]
            # Extrai apenas o nome da variável (última parte após o último ponto)
            var_name = full_path.split('.')[-1]
            tag_mapping[primary_key] = var_name
    
    return tag_mapping

def transform_data_to_ml_format(csv_path, config_path, output_path=None):
    """
    Transforma os dados do formato vertical (uma linha por medição)
    para formato horizontal (uma linha por timestamp com todas as variáveis)
    """
    print("Carregando configuração de tags...")
    tag_mapping = load_tag_config(config_path)
    print(f"Tags mapeadas: {tag_mapping}")
    
    print("\nCarregando dados do CSV...")
    df = pd.read_csv(csv_path)
    
    # Adiciona o nome da variável com base no PrimaryKey
    df['VariableName'] = df['PrimaryKey'].map(tag_mapping)
    
    # Converte valores booleanos (true/false) para numéricos (1/0)
    df['Value'] = df['Value'].apply(lambda x: 1 if x == 'true' else (0 if x == 'false' else x))
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Converte timestamp para datetime
    df['SourceTimeStamp'] = pd.to_datetime(df['SourceTimeStamp'])
    
    print(f"\nTotal de registros originais: {len(df)}")
    print(f"Variáveis únicas: {df['VariableName'].nunique()}")
    print(f"Timestamps únicos: {df['SourceTimeStamp'].nunique()}")
    
    # Pivota os dados: cada linha é um timestamp, cada coluna é uma variável
    print("\nPivotando dados...")
    df_pivot = df.pivot_table(
        index='SourceTimeStamp',
        columns='VariableName',
        values='Value',
        aggfunc='first'  # Se houver múltiplos valores no mesmo timestamp, pega o primeiro
    )
    
    # Reseta o índice para ter SourceTimeStamp como coluna
    df_pivot = df_pivot.reset_index()
    
    # Ordena por timestamp
    df_pivot = df_pivot.sort_values('SourceTimeStamp')
    
    # Preenche valores faltantes com forward fill (propaga o último valor conhecido)
    df_pivot = df_pivot.fillna(method='ffill')
    
    # Remove linhas onde ainda há valores nulos (início do dataset)
    df_pivot = df_pivot.dropna()

    # round all numeric columns to 5 decimal places
    numeric_cols = df_pivot.select_dtypes(include=['float64', 'int64']).columns
    df_pivot[numeric_cols] = df_pivot[numeric_cols].round(5)
    
    # Adiciona metadados do cliente e falhas
    df_pivot['client_id'] = CLIENT_ID
    df_pivot['has_fault'] = int(HAS_FAULT)
    
    print(f"\nDataset transformado:")
    print(f"  - Linhas (timestamps): {len(df_pivot)}")
    print(f"  - Colunas (variáveis + timestamp): {len(df_pivot.columns)}")
    print(f"  - Client ID: {CLIENT_ID}")
    print(f"  - Tem falha: {HAS_FAULT}")
    print(f"\nPrimeiras linhas:")
    print(df_pivot.head())
    
    print(f"\nInformações sobre as colunas:")
    print(df_pivot.info())
    
    # Salva o resultado
    if output_path is None:
        output_path = csv_path.replace('.csv', '_transformed.csv')
    
    df_pivot.to_csv(output_path, index=False)
    print(f"\nDataset salvo em: {output_path}")
    
    return df_pivot

if __name__ == "__main__":
    # Caminhos dos arquivos
    base_path = Path(__file__).parent
    csv_path = base_path / "data" / CSV_FILE_NAME
    config_path = base_path / "data" / "tagconfig.txt"
    output_path = base_path / "data" / f"{CSV_FILE_NAME.split('.')[0]}_transformed.csv"
    
    # Executa a transformação
    df_transformed = transform_data_to_ml_format(
        str(csv_path),
        str(config_path),
        str(output_path)
    )
