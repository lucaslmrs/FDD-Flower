import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURAÇÕES MANUAIS - ALTERE AQUI
# =============================================================================

# Nome do arquivo CSV a ser processado (deve estar na pasta 'data')
CSV_FILE_NAME = "cenario14.csv"  # TODO: Altere para o nome do arquivo desejado

# Identificador do cliente/cenário
CLIENT_ID = "B"  # TODO: Altere para o ID do cliente correspondente

# Flag indicando se este cenário contém falhas
HAS_FAULT = 1  # TODO: Altere para True se o cenário contém falhas

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

def transform_data_to_ml_format(csv_path, config_path, use_forward_fill=True, output_path=None):
    """
    Transforma os dados do formato vertical (uma linha por medição)
    para formato horizontal (uma linha por timestamp com todas as variáveis)
    
    Args:
        csv_path: Caminho do arquivo CSV de entrada
        config_path: Caminho do arquivo de configuração de tags
        use_forward_fill: Se True, preenche valores faltantes com forward fill
        output_path: Caminho do arquivo de saída
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
    
    if use_forward_fill:
        print("Aplicando forward fill para preencher valores faltantes...")
        # Preenche valores faltantes com forward fill (propaga o último valor conhecido)
        df_pivot = df_pivot.ffill()
        
        # Remove linhas onde ainda há valores nulos (início do dataset)
        df_pivot = df_pivot.dropna()
    else:
        print("Mantendo valores faltantes (NaN) onde não há dados...")
    
    # Arredonda todas as colunas numéricas para 5 casas decimais
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
    
    if not use_forward_fill:
        null_counts = df_pivot.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\nValores nulos por coluna:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"  - {col}: {count} valores nulos")
    
    print(f"\nPrimeiras linhas:")
    print(df_pivot.head(10))
    
    print(f"\nInformações sobre as colunas:")
    print(df_pivot.info())
    
    # Salva o resultado
    if output_path is None:
        suffix = '_with_ffill.csv' if use_forward_fill else '_without_ffill.csv'
        output_path = csv_path.replace('.csv', suffix)
    
    df_pivot.to_csv(output_path, index=False)
    print(f"\nDataset salvo em: {output_path}")
    
    return df_pivot

if __name__ == "__main__":

    # Caminhos dos arquivos
    base_path = Path(__file__).parent
    csv_path = base_path / "data" / CSV_FILE_NAME
    config_path = base_path / "data" / "tagconfig.txt"
    
    print("="*60)
    print("VERSÃO 1: COM FORWARD FILL")
    print("="*60)
    output_path_with_ffill = base_path / "data" / f"{CSV_FILE_NAME.split('.')[0]}_with_ffill.csv"
    df_with_ffill = transform_data_to_ml_format(
        str(csv_path),
        str(config_path),
        use_forward_fill=True,
        output_path=str(output_path_with_ffill)
    )
    
    print("\n\n")
    print("="*60)
    print("VERSÃO 2: SEM FORWARD FILL (mantém valores NaN)")
    print("="*60)
    output_path_without_ffill = base_path / "data" / f"{CSV_FILE_NAME.split('.')[0]}_without_ffill.csv"
    df_without_ffill = transform_data_to_ml_format(
        str(csv_path),
        str(config_path),
        use_forward_fill=False,
        output_path=str(output_path_without_ffill)
    )
    
    print("\n\n")
    print("="*60)
    print("TRANSFORMAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print(f"\n📊 Dois arquivos foram gerados:")
    print(f"   1. {output_path_with_ffill.name} - COM forward fill (sem valores nulos)")
    print(f"   2. {output_path_without_ffill.name} - SEM forward fill (com valores nulos onde não há dados)")
    print("\n💡 Use a versão COM forward fill quando quiser dados completos.")
    print("💡 Use a versão SEM forward fill quando quiser ver exatamente quando cada medição foi feita.")
