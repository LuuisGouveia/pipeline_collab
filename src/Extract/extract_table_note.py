import pandas as pd
import numpy as np
import re 
import os 

def extract_table_note(file_path):
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")

    match_data = re.search(r'(\d{2}/\d{2}/\d{4})', file_name)
    if match_data:
        data_final_str = f"{match_data.group(3)}-{match_data.group(2)}-{match_data.group(1)}"
        print(f"Extracted date: {data_final_str}")
    else:
        if 'Carlos' in file_name and "2020" in file_name:
                  data_final_str = "2020-04-08"
        else:
            data_final_str = "2026-01-01"
        print(f"[DATA] Não encontrada no nome. Aplicando fallback: {data_final_str}")

    df_raw = pd.read_csv(file_path, header=None)
    print(f"Raw data shape: {df_raw.shape}")

    client_name = None
    line_header_index = None
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
        
        columns = [str(c).strip() for c in df_raw.iloc[line_header_index].dropna()]

        df_data = df_raw.iloc[line_header_index + 1:].copy()
        df_data = df_data.iloc[:,:len(columns)]
        df_data.columns = columns

        df_data.df_data.replace(r'^\s*$', np.nan, regex=True)
        df_data.dropna(subset=['Trabalho'], inplace=True)
        df_data['Paciente'] = df_data['Paciente'].ffill()

        df_data = df_data[~df_data['Paciente'].astype(str).str.contains('TOTAL | Débito', case=False, na=False)]
        df_data = df_data[~df_data['Trabalho'].astype(str).str.contains('TOTAL | Débito', case=False, na=False)]
        def clean_values(v):
            if pd.isna(v): return 0.0
            s = str(v).strip().replace('R$', '').strip()
            if s.isdigit(): return float(s)
            if ',' in s: s= s.replace('.', '').replace(',','.')
            try: return float(s)
            except: return 0.0
        
        for col in ['Vlr Unit', 'Vlr Total','VALOR','Valor Unitário', 'Valor Total']:
            if col in df_data.columns:
                df_data[col] = df_data[col].apply(clean_values) 
        
        df_final = pd.DataFrame(index=df_data.index)
        df_final['data_entrega'] = pd.to_datetime(data_final_str)
        df_final['cliente_dentista'] = client_name
        df_final['paciente'] = df_data['Paciente']
        df_final['servico_trabalho'] = df_data['Trabalho']

        col_q = 'Quant.' if 'Quant.' in df_data.columns else 'QUANT' if 'QUANT' in df_data.columns else None
        df_final['quantidade'] = pd.to_numeric(df_data[col_q], errors='coerce').fillna(1).astype(int) if col_q else 1
        df_final['valor_total'] = df_data['Vlr Total'] if 'Vlr Total' in df_data.columns else df_data['Valor Total'] if 'Valor Total' in df_data.columns else 0.0
        df_final['valor_unitario'] = df_final['valor_total'] / df_final['quantidade'] 
        
        df_final.reset_index(drop=True, inplace=True)
        df_final.drop_duplicates(inplace=True)
        
        return df_final
    
