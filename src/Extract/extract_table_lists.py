import os
import re
import numpy as np
import pandas as pd

def extract_data_table_list(file_path):
    """
    Processa um arquivo individual de nota de serviço (formato CSV/tabular),
    corrige inconsistências de colunas e retorna um DataFrame estruturado e limpo.
    """
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")
    
    try:
        df_raw = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o arquivo {file_name}: {e}")
        return None
        
    print(f"Raw data shape: {df_raw.shape}")

    # Correção do split vazio: divide o nome do arquivo usando espaços ou hifens
    # para obter o primeiro termo (geralmente o nome do cliente)
    name_parts = re.split(r'[\s\-_]+', file_name)
    client_name = name_parts[0] if name_parts else "Desconhecido"
    print(f"Extracted client name: {client_name}")

    header_line_index = None
    for idx, row in df_raw.iterrows():
        line_values = row.dropna().astype(str).str.strip().tolist()
        line_text_unique = ' '.join(line_values)
        if 'paciente' in line_text_unique.lower() and 'trabalho' in line_text_unique.lower():
            header_line_index = idx
            print(f"Header line found at index: {header_line_index}")
            break
            
    if header_line_index is None:
        print(f"[ERRO] Cabeçalho da tabela de serviços não localizado em {file_name}")
        return None

    # Normaliza as colunas em CAIXA ALTA para evitar KeyErrors devido à variação de escrita
    columns = [str(x).strip().upper() for x in df_raw.iloc[header_line_index].dropna()]

    # Filtra as linhas de dados pulando o cabeçalho
    df_data = df_raw.iloc[header_line_index + 1:].copy()
    df_data = df_data.iloc[:, :len(columns)]
    df_data.columns = columns

    # Corrige a sintaxe de substituição de strings vazias por NaN
    df_data = df_data.replace(r'^\s*$', np.nan, regex=True)
    
    # Identifica a coluna de trabalho de forma segura
    work_col = 'DESCRIÇÃO DO TRABALHO' if 'DESCRIÇÃO DO TRABALHO' in df_data.columns else 'TRABALHO' if 'TRABALHO' in df_data.columns else None
    
    if work_col:
        df_data.dropna(subset=[work_col], inplace=True)
    
    if 'PACIENTE' in df_data.columns:
        # Descarta linhas repetidas que possam conter a palavra-chave "paciente"
        df_data = df_data[df_data['PACIENTE'].astype(str).str.strip().str.lower() != 'paciente']
        df_data['PACIENTE'] = df_data['PACIENTE'].fillna('Desconhecido')
        df_data['PACIENTE'] = df_data['PACIENTE'].ffill()

    # Limpa linhas contendo somatórias ou débitos
    if 'PACIENTE' in df_data.columns:
        df_data = df_data[~df_data['PACIENTE'].astype(str).str.contains('TOTAL | Débito', case=False, na=False)]
    if work_col:
        df_data = df_data[~df_data[work_col].astype(str).str.contains('TOTAL | Débito', case=False, na=False)]

    def clean_values(v):
        if pd.isna(v): 
            return 0.0
        s = str(v).strip().replace('R$', '').replace('$', '').strip()
        if s.isdigit(): 
            return float(s)
        if ',' in s: 
            s = s.replace('.', '').replace(',', '.')
        try: 
            return float(s)
        except: 
            return 0.0
    
    # Trata as colunas financeiras mapeadas para caixa alta
    for col in ['VLR UNIT', 'VLR TOTAL', 'VALOR', 'VALOR TOTAL', 'VALOR UNITÁRIO']:
        if col in df_data.columns:
            df_data[col] = df_data[col].apply(clean_values)

    df_final = pd.DataFrame(index=df_data.index)

    # Coleta a data de entrega
    date_col = 'DATA ENTREGA' if 'DATA ENTREGA' in df_data.columns else 'ENTREGA' if 'ENTREGA' in df_data.columns else None
    if date_col:
        df_final['data_entrega'] = pd.to_datetime(df_data[date_col], errors='coerce')
    else:
        df_final['data_entrega'] = pd.NaT

    df_final['cliente_dentista'] = client_name
    df_final['paciente'] = df_data['PACIENTE'] if 'PACIENTE' in df_data.columns else 'Desconhecido'
    df_final['servico_trabalho'] = df_data[work_col] if work_col else 'Não especificado'

    # Quantidade
    col_q = 'QUANT.' if 'QUANT.' in df_data.columns else 'QUANT' if 'QUANT' in df_data.columns else 'QUANTIDADE' if 'QUANTIDADE' in df_data.columns else None
    df_final['quantidade'] = pd.to_numeric(df_data[col_q], errors='coerce').fillna(1).astype(int) if col_q else 1
    
    # Valor total e unitário
    val_total_col = 'VLR TOTAL' if 'VLR TOTAL' in df_data.columns else 'VALOR' if 'VALOR' in df_data.columns else 'VALOR TOTAL' if 'VALOR TOTAL' in df_data.columns else None
    df_final['valor_total'] = df_data[val_total_col] if val_total_col else 0.0
    df_final['valor_unitario'] = df_final['valor_total'] / df_final['quantidade']

    df_final.reset_index(drop=True, inplace=True)
    df_final.drop_duplicates(inplace=True)
    
    return df_final

def extract_table_lists(folder_path):
    """
    Varre toda a pasta especificada processando cada arquivo CSV encontrado
    e retorna um DataFrame consolidado contendo todos os dados extraídos.
    """
    all_dfs = []
    
    if not os.path.exists(folder_path):
        print(f"[AVISO] Pasta '{folder_path}' não foi localizada.")
        return pd.DataFrame()
        
    # Filtra apenas os arquivos CSV da pasta
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    
    if not files:
        print(f"[AVISO] Nenhum arquivo CSV encontrado na pasta '{folder_path}'.")
        return pd.DataFrame()
        
    total_files = len(files)
    print(f"Iniciando varredura em lote: {total_files} arquivos encontrados.")
    
    for idx, filename in enumerate(files, start=1):
        file_path = os.path.join(folder_path, filename)
        print(f"\n[{idx}/{total_files}] Processando arquivo...")
        
        try:
            df_file = extract_data_table_list(file_path)
            if df_file is not None and not df_file.empty:
                df_file['arquivo_origem'] = filename
                all_dfs.append(df_file)
        except Exception as e:
            print(f"[ERRO] Falha crítica ao processar {filename}: {e}")
            
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nProcessamento concluído com sucesso! Total de {len(combined_df)} linhas consolidadas.")
        return combined_df
    else:
        print("\n[AVISO] Nenhum dado válido pôde ser extraído dos arquivos.")
        return pd.DataFrame()
