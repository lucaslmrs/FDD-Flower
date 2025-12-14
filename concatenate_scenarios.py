"""
Concatena todos os cenários com sufixo _with_ffill.csv em um único dataset.

Este script procura todos os arquivos CSV com sufixo '_with_ffill.csv' na pasta 'data'
e os concatena em um único arquivo, preservando as colunas de metadados (client_id, has_fault).
"""

import pandas as pd
from pathlib import Path
import glob

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Pasta onde estão os arquivos
DATA_FOLDER = "data"

# Padrão dos arquivos a serem concatenados
FILE_PATTERN = "*_with_ffill.csv"

# Nome do arquivo de saída
OUTPUT_FILE = "all_scenarios_concatenated.csv"

# =============================================================================


def concatenate_scenarios(data_folder: str = DATA_FOLDER, 
                         file_pattern: str = FILE_PATTERN,
                         output_file: str = OUTPUT_FILE) -> pd.DataFrame:
    """
    Concatena todos os arquivos CSV que correspondem ao padrão especificado.
    
    Args:
        data_folder: Pasta onde estão os arquivos CSV
        file_pattern: Padrão glob para encontrar os arquivos (ex: '*_with_ffill.csv')
        output_file: Nome do arquivo de saída
        
    Returns:
        DataFrame concatenado com todos os cenários
    """
    base_path = Path(__file__).parent
    data_path = base_path / data_folder
    
    # Encontrar todos os arquivos que correspondem ao padrão
    search_pattern = str(data_path / file_pattern)
    csv_files = sorted(glob.glob(search_pattern))
    
    if not csv_files:
        print(f"⚠️  Nenhum arquivo encontrado com o padrão: {file_pattern}")
        print(f"   Pasta de busca: {data_path}")
        return None
    
    print(f"📂 Encontrados {len(csv_files)} arquivos para concatenar:")
    for i, file in enumerate(csv_files, 1):
        file_name = Path(file).name
        print(f"   {i}. {file_name}")
    
    print("\n🔄 Carregando e concatenando arquivos...")
    
    # Carregar e concatenar todos os DataFrames
    dataframes = []
    total_rows = 0
    
    for file in csv_files:
        df = pd.read_csv(file)
        rows = len(df)
        total_rows += rows
        dataframes.append(df)
        
        file_name = Path(file).name
        client_id = df['client_id'].iloc[0] if 'client_id' in df.columns else 'N/A'
        has_fault = df['has_fault'].iloc[0] if 'has_fault' in df.columns else 'N/A'
        
        print(f"   ✓ {file_name}: {rows} linhas (client_id={client_id}, has_fault={has_fault})")
    
    # Concatenar todos os DataFrames
    df_concatenated = pd.concat(dataframes, ignore_index=True)
    
    # Estatísticas
    print(f"\n📊 Estatísticas do dataset concatenado:")
    print(f"   - Total de linhas: {len(df_concatenated)}")
    print(f"   - Total de colunas: {len(df_concatenated.columns)}")
    
    if 'client_id' in df_concatenated.columns:
        unique_clients = df_concatenated['client_id'].nunique()
        print(f"   - Clientes únicos: {unique_clients}")
        print(f"   - Distribuição por cliente:")
        client_counts = df_concatenated['client_id'].value_counts().sort_index()
        for client_id, count in client_counts.items():
            print(f"     • Client {client_id}: {count} linhas")
    
    if 'has_fault' in df_concatenated.columns:
        fault_counts = df_concatenated['has_fault'].value_counts()
        print(f"   - Distribuição de falhas:")
        print(f"     • Sem falha (0): {fault_counts.get(0, 0)} linhas")
        print(f"     • Com falha (1): {fault_counts.get(1, 0)} linhas")
    
    # Salvar arquivo concatenado
    output_path = base_path / data_folder / output_file
    df_concatenated.to_csv(output_path, index=False)
    
    print(f"\n✅ Arquivo concatenado salvo em: {output_path}")
    print(f"   Tamanho: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return df_concatenated


if __name__ == "__main__":
    print("="*70)
    print("  CONCATENAÇÃO DE CENÁRIOS")
    print("="*70)
    print()
    
    df_result = concatenate_scenarios()
    
    if df_result is not None:
        print("\n" + "="*70)
        print("  ✓ CONCATENAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*70)
        
        print("\n💡 Dica: O arquivo concatenado pode ser usado para:")
        print("   - Treinamento de modelos com múltiplos cenários")
        print("   - Análise comparativa entre diferentes clientes")
        print("   - Detecção de anomalias cross-scenario")
