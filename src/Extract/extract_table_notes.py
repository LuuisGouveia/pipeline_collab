import os
import re
import numpy as np
import pandas as pd

def extract_data_table_note(file_path):
    """
    Processa um arquivo individual de nota de serviço (formato CSV/tabular)
    e retorna um DataFrame estruturado com os dados limpos.
    """
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")

    # Regex robusto para capturar Dia, Mês e Ano (aceita espaços, barras ou traços)
    match_data = re.search(r'(\d{2})[\s\-_/](\d{2})[\s\-_/](\d{2,4})', file_name)
    if match_data:
        day, month, year = match_data.groups()
        if len(year) == 2:
            year = "20" + year
        data_final_str = f"{year}-{month}-{day}"
        print(f"Extracted date: {data_final_str}")
    else:
        if 'Carlos' in file_name and "2020" in file_name:
            data_final_str = "2020-04-08"
        else:
            data_final_str = "2026-01-01"
        print(f"[DATA] Não encontrada no nome. Aplicando fallback: {data_final_str}")

    try:
        df_raw = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o arquivo {file_name}: {e}")
        return None

    print(f"Raw data shape: {df_raw.shape}")

    client_name = None
    line_header_index = None
    
    # Procura a linha de cabeçalho e extrai o nome do cliente
    for idx, row in df_raw.iterrows():
        line_values = row.dropna().astype(str).str.strip().tolist()
        line_text_unique = ' '.join(line_values)
        
        if 'paciente' in line_text_unique.lower() and 'trabalho' in line_text_unique.lower():
            line_header_index = idx
            print(f"Header line found at index: {line_header_index}")
            break
            
        if not client_name:
            match_client = re.search(r'([A-Za-z]+)', line_text_unique)
            if match_client:
                client_name = match_client.group(1).strip()
                print(f"Extracted client name: {client_name}")
        
        if 'Paciente' in line_text_unique and 'Trabalho' in line_text_unique:
            line_header_index = idx
            print(f"Header line found at index: {line_header_index}")
            break

    if line_header_index is None:
        print(f"[ERRO] Tabela de serviços não localizada em {file_name}")
        return None
        
    # Extrai colunas e filtra linhas de dados
    columns = [str(c).strip() for c in df_raw.iloc[line_header_index].dropna()]

    df_data = df_raw.iloc[line_header_index + 1:].copy()
    df_data = df_data.iloc[:, :len(columns)]
    df_data.columns = columns

    # Correção da substituição de strings vazias por NaN
    df_data = df_data.replace(r'^\s*$', np.nan, regex=True)
    
    # Remove linhas sem descrição de Trabalho
    if 'Trabalho' in df_data.columns:
        df_data.dropna(subset=['Trabalho'], inplace=True)
    
    if 'Paciente' in df_data.columns:
        df_data['Paciente'] = df_data['Paciente'].ffill()

    # Limpa linhas contendo somatórias ou débitos
    if 'Paciente' in df_data.columns:
        df_data = df_data[~df_data['Paciente'].astype(str).str.contains('TOTAL | Débito', case=False, na=False)]
    if 'Trabalho' in df_data.columns:
        df_data = df_data[~df_data['Trabalho'].astype(str).str.contains('TOTAL | Débito', case=False, na=False)]
        
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
        
    for col in ['Vlr Unit', 'Vlr Total', 'VALOR', 'Valor Unitário', 'Valor Total']:
        if col in df_data.columns:
            df_data[col] = df_data[col].apply(clean_values) 
        
    df_final = pd.DataFrame(index=df_data.index)
    df_final['data_entrega'] = pd.to_datetime(data_final_str)
    df_final['cliente_dentista'] = client_name
    df_final['paciente'] = df_data['Paciente'] if 'Paciente' in df_data.columns else 'Não especificado'
    df_final['servico_trabalho'] = df_data['Trabalho'] if 'Trabalho' in df_data.columns else 'Não especificado'

    col_q = 'Quant.' if 'Quant.' in df_data.columns else 'QUANT' if 'QUANT' in df_data.columns else None
    df_final['quantidade'] = pd.to_numeric(df_data[col_q], errors='coerce').fillna(1).astype(int) if col_q else 1
    
    val_total_col = 'Vlr Total' if 'Vlr Total' in df_data.columns else 'Valor Total' if 'Valor Total' in df_data.columns else None
    df_final['valor_total'] = df_data[val_total_col] if val_total_col else 0.0
    df_final['valor_unitario'] = df_final['valor_total'] / df_final['quantidade'] 
        
    df_final.reset_index(drop=True, inplace=True)
    df_final.drop_duplicates(inplace=True)
        
    return df_final

def extract_table_notes(folder_path):
    """
    Varre toda a pasta especificada processando cada arquivo CSV encontrado
    e retorna um DataFrame consolidado contendo todos os dados extraídos.
    """
    all_dfs = []
    
    if not os.path.exists(folder_path):
        print(f"[AVISO] Pasta '{folder_path}' não foi localizada.")
        return pd.DataFrame()
        
    # Localiza apenas arquivos de dados (CSV) na pasta
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
            df_file = extract_data_table_note(file_path)
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

