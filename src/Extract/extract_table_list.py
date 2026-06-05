import pandas as pd
import numpy as np
import re
import os

def extract_table_list(file_path):
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")
    df_raw = pd.read_csv(file_path, header=None)
    print(f"Raw data shape: {df_raw.shape}")

    client_name = file_name.split('')[0]
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
        print("Header line not found. Please check the file format.")
        return None
    columns = [str(x).strip().lower() for x in df_raw.iloc[header_line_index].dropna()]

    df_data = df_raw.replace(r'^\s*$', np.nan, regex=True)
    df_data.dropna(subset=['DESCRIÇÃO DO TRABALHO'], inplace=True)
    
    if 'PACIENTE' in df_data.columns:
        df_data = df_data[df_data['PACIENTE'].astype(str).str.strip().str.lower() != 'paciente']
    
    df_data['PACIENTE'] = df_data['PACIENTE'].fillna('Desconhecido')

    df_data = df_data[~df_data['PACIENTE'].astype(str).str.contains('TOTAL | Débito', case=False, na=False)]
    df_data = df_data[~df_data['DESCRIÇÃO DO TRABALHO'].astype(str).str.contains('TOTAL | Débito', case=False, na=False)]

    def clean_values(v):
        if pd.isna(v): return 0.0
        s = str(v).strip().replace('R$', '').strip()
        if s.isdigit(): return float(s)
        if ',' in s: s= s.replace('.', '').replace(',','.')
        try: return float(s)
        except: return 0.0
    
    for col in ['Vlr Unit', 'Vlr Total','VALOR']:
        if col in df_data.columns:
            df_data[col] = df_data[col].apply(clean_values)

    df_final = pd.DataFrame(index=df_data.index)

    df_final['data_entrega'] = df_data['DATA ENTREGA'] if 'DATA ENTREGA' in df_data.columns else df_data['Entrega'] if 'Entrega' in df_data.columns else pd.NaT
    df_final['cliente_dentista'] = client_name
    df_final['paciente'] = df_data['PACIENTE']
    df_final['servico_trabalho'] = df_data['DESCRIÇÃO DO TRABALHO']

    col_q = 'Quant.' if 'Quant.' in df_data.columns else 'QUANT'
    df_final['quantidade'] = pd.to_numeric(df_data[col_q], errors='coerce').fillna(1).astype(int)
    df_final['valor_total'] = df_data['Vlr Total'] if 'Vlr Total' in df_data.columns else df_data['VALOR']
    df_final['valor_unitario'] = df_final['valor_total'] / df_final['quantidade']

    df_final.reset_index(drop=True, inplace=True)
    df_final.drop_duplicates(inplace=True)
    return df_final

