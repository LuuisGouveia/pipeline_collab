import os
import numpy as np
import pandas as pd
from pypdf import PdfReader
import utils


# ==============================================================================
# BLOCO 1: PROCESSADOR DE NOTAS DE SERVIÇO (CSV TABULAR)
# ==============================================================================
def extract_data_table_note(file_path):
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")

    # Uso do utilitário de data compartilhado
    data_final_str = (
        utils.extract_date_from_filename(file_name, fallback_carlos=True)
        or "2026-01-01"
    )

    try:
        df_raw = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o arquivo {file_name}: {e}")
        return None

    line_header_index, client_name = None, None

    for idx, row in df_raw.iterrows():
        line_text_unique = " ".join(row.dropna().astype(str).str.strip().tolist())

        if (
            "paciente" in line_text_unique.lower()
            and "trabalho" in line_text_unique.lower()
        ):
            line_header_index = idx
            break
        if (
            "cliente" in line_text_unique.lower()
            or "dentista" in line_text_unique.lower()
        ):
            import re

            match_client = re.search(
                r"(?:cliente|dentista)\s*:?\s*(.+)", line_text_unique, re.IGNORECASE
            )
            if match_client:
                client_name = (
                    match_client.group(1).replace('"', "").replace("'", "").strip()
                )

    if not client_name and df_raw.shape[0] > 0:
        import re

        match_client = re.search(
            r"([A-Za-z]+)", " ".join(df_raw.iloc[0].dropna().astype(str))
        )
        if match_client:
            client_name = match_client.group(1).strip()

    if line_header_index is None:
        return None

    columns = [str(c).strip() for c in df_raw.iloc[line_header_index].dropna()]
    df_data = df_raw.iloc[line_header_index + 1 :].copy().iloc[:, : len(columns)]
    df_data.columns = columns
    df_data.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    if "Trabalho" in df_data.columns:
        df_data.dropna(subset=["Trabalho"], inplace=True)
    if "Paciente" in df_data.columns:
        df_data["Paciente"] = df_data["Paciente"].ffill()

    # Filtros de totais e débitos
    for col in ["Paciente", "Trabalho"]:
        if col in df_data.columns:
            df_data = df_data[
                ~df_data[col]
                .astype(str)
                .str.contains("TOTAL | Débito", case=False, na=False)
            ]

    # Uso do utilitário financeiro compartilhado
    for col in ["Vlr Unit", "Vlr Total", "VALOR", "Valor Unitário", "Valor Total"]:
        if col in df_data.columns:
            df_data[col] = df_data[col].apply(utils.clean_monetary_values)

    df_final = pd.DataFrame(index=df_data.index)
    df_final["data_entrega"] = pd.to_datetime(data_final_str)
    df_final["cliente_dentista"] = client_name
    df_final["paciente"] = (
        df_data["Paciente"] if "Paciente" in df_data.columns else "Não especificado"
    )
    df_final["servico_trabalho"] = (
        df_data["Trabalho"] if "Trabalho" in df_data.columns else "Não especificado"
    )

    col_q = (
        "Quant."
        if "Quant." in df_data.columns
        else "QUANT" if "QUANT" in df_data.columns else None
    )
    df_final["quantidade"] = (
        pd.to_numeric(df_data[col_q], errors="coerce").fillna(1).astype(int)
        if col_q
        else 1
    )

    val_total_col = (
        "Vlr Total"
        if "Vlr Total" in df_data.columns
        else "Valor Total" if "Valor Total" in df_data.columns else None
    )
    df_final["valor_total"] = df_data[val_total_col] if val_total_col else 0.0
    df_final["valor_unitario"] = df_final["valor_total"] / df_final["quantidade"]

    df_final.reset_index(drop=True, inplace=True)
    return df_final.drop_duplicates()


# ==============================================================================
# BLOCO 2: PROCESSADOR DE RELATÓRIOS INTELIGENTES (PDF VIA HUGGING FACE)
# ==============================================================================
def extract_pdf_text(pdf_path):
    full_text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    except Exception as e:
        print(f"Erro ao ler o PDF {pdf_path}: {e}")
    return full_text


def process_single_pdf(source_dir, filename):
    full_path = os.path.join(source_dir, filename)
    filename_date_fallback = utils.extract_date_from_filename(filename)
    raw_text = extract_pdf_text(full_path)

    if not raw_text.strip():
        return []

    json_data = utils.structure_data_with_huggingface(raw_text)
    services = []
    for service in json_data.get("services", []):
        final_date = service.get("delivery_date")
        if not final_date or final_date.strip() == "":
            final_date = (
                filename_date_fallback if filename_date_fallback else "Não especificada"
            )

        service["final_delivery_date"] = final_date
        service["source_file"] = filename
        services.append(service)
    return services


# ==============================================================================
# BLOCO 3: PROCESSADOR DE LISTAS DE SERVIÇO (CSV COMPLEXO)
# ==============================================================================
def extract_data_table_list(file_path):
    file_name = os.path.basename(file_path)
    print(f"Processing file: {file_name}")

    try:
        df_raw = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o arquivo {file_name}: {e}")
        return None

    import re

    name_parts = re.split(r"[\s\-_]+", file_name)
    client_name = name_parts[0] if name_parts else "Desconhecido"

    header_line_index = None
    for idx, row in df_raw.iterrows():
        line_text_unique = " ".join(row.dropna().astype(str).str.strip().tolist())
        if (
            "paciente" in line_text_unique.lower()
            and "trabalho" in line_text_unique.lower()
        ):
            header_line_index = idx
            break

    if header_line_index is None:
        return None

    columns = [str(x).strip().upper() for x in df_raw.iloc[header_line_index].dropna()]
    df_data = df_raw.iloc[header_line_index + 1 :].copy().iloc[:, : len(columns)]
    df_data.columns = columns
    df_data.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    work_col = (
        "DESCRIÇÃO DO TRABALHO"
        if "DESCRIÇÃO DO TRABALHO" in df_data.columns
        else "TRABALHO" if "TRABALHO" in df_data.columns else None
    )
    if work_col:
        df_data.dropna(subset=[work_col], inplace=True)

    if "PACIENTE" in df_data.columns:
        df_data = df_data[
            df_data["PACIENTE"].astype(str).str.strip().str.lower() != "paciente"
        ]
        df_data["PACIENTE"] = df_data["PACIENTE"].fillna("Desconhecido").ffill()

    for col in ["PACIENTE", work_col]:
        if col and col in df_data.columns:
            df_data = df_data[
                ~df_data[col]
                .astype(str)
                .str.contains("TOTAL | Débito", case=False, na=False)
            ]

    # Uso do utilitário financeiro compartilhado
    for col in ["VLR UNIT", "VLR TOTAL", "VALOR", "VALOR TOTAL", "VALOR UNITÁRIO"]:
        if col in df_data.columns:
            df_data[col] = df_data[col].apply(utils.clean_monetary_values)

    df_final = pd.DataFrame(index=df_data.index)
    # Coleta a data de entrega interna da tabela
    date_col = (
        "DATA ENTREGA"
        if "DATA ENTREGA" in df_data.columns
        else "ENTREGA" if "ENTREGA" in df_data.columns else None
    )
    if date_col:
        df_final["data_entrega"] = pd.to_datetime(df_data[date_col], errors="coerce")
    else:
        df_final["data_entrega"] = pd.NaT

    # --- 🌟 CORREÇÃO / FALLBACK INTELIGENTE 🌟 ---
    # Se a coluna não existia ou se os valores internos vieram nulos (NaT), busca no nome do arquivo
    filename_date = utils.extract_date_from_filename(file_name)
    if filename_date:
        df_final["data_entrega"] = df_final["data_entrega"].fillna(
            pd.to_datetime(filename_date)
        )

    df_final["cliente_dentista"] = client_name
    df_final["paciente"] = (
        df_data["PACIENTE"] if "PACIENTE" in df_data.columns else "Desconhecido"
    )
    df_final["servico_trabalho"] = df_data[work_col] if work_col else "Não especificado"

    col_q = (
        "QUANT."
        if "QUANT." in df_data.columns
        else (
            "QUANT"
            if "QUANT" in df_data.columns
            else "QUANTIDADE" if "QUANTIDADE" in df_data.columns else None
        )
    )
    df_final["quantidade"] = (
        pd.to_numeric(df_data[col_q], errors="coerce").fillna(1).astype(int)
        if col_q
        else 1
    )

    val_total_col = (
        "VLR TOTAL"
        if "VLR TOTAL" in df_data.columns
        else (
            "VALOR"
            if "VALOR" in df_data.columns
            else "VALOR TOTAL" if "VALOR TOTAL" in df_data.columns else None
        )
    )
    df_final["valor_total"] = df_data[val_total_col] if val_total_col else 0.0
    df_final["valor_unitario"] = df_final["valor_total"] / df_final["quantidade"]

    df_final.reset_index(drop=True, inplace=True)
    return df_final.drop_duplicates()


# ==============================================================================
# ORQUESTRAÇÃO DE LOTES (Geral para os 3 tipos)
# ==============================================================================
def batch_process_folder(folder_path, process_type, parquet_path=None, delay=0.0):
    """Orquestrador genérico que varre a pasta aplicando a função adequada"""
    if not os.path.exists(folder_path):
        print(f"[AVISO] Pasta '{folder_path}' não localizada.")
        return pd.DataFrame()

    ext = ".pdf" if process_type == "pdf" else ".csv"
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext)]

    if not files:
        print(f"[AVISO] Nenhum arquivo {ext} encontrado em '{folder_path}'.")
        return pd.DataFrame()

    all_data = []
    print(
        f"Iniciando varredura em lote ({process_type.upper()}): {len(files)} arquivos."
    )

    for idx, filename in enumerate(files, start=1):
        file_path = os.path.join(folder_path, filename)
        print(f"\n[{idx}/{len(files)}] Processando: {filename}")

        try:
            if process_type == "note_csv":
                df_file = extract_data_table_note(file_path)
                if df_file is not None and not df_file.empty:
                    df_file["arquivo_origem"] = filename
                    all_data.append(df_file)

            elif process_type == "list_csv":
                df_file = extract_data_table_list(file_path)
                if df_file is not None and not df_file.empty:
                    df_file["arquivo_origem"] = filename
                    all_data.append(df_file)

            elif process_type == "pdf":
                import time

                services = process_single_pdf(folder_path, filename)
                if services:
                    all_data.extend(services)
                if idx < len(files) and delay > 0:
                    time.sleep(delay)

        except Exception as e:
            print(f"[ERRO] Falha crítica ao processar {filename}: {e}")

    if not all_data:
        return pd.DataFrame()

    # Consolidação final do DataFrame
    if process_type == "pdf":
        df = pd.DataFrame(all_data)
        ordered_columns = [
            "dentist_client",
            "patient",
            "work_performed",
            "unit_value",
            "quantity",
            "total_value",
            "final_delivery_date",
            "source_file",
        ]
        df = df[ordered_columns]
        df.columns = [
            "Cliente (Dentista)",
            "Paciente",
            "Trabalho Executado",
            "Valor Unitário",
            "Quantidade",
            "Valor Total",
            "Data de Entrega",
            "Arquivo de Origem",
        ]
    else:
        df = pd.concat(all_data, ignore_index=True)

    print(f"\nConcluído! Total de {len(df)} linhas consolidadas.")
    utils.save_to_parquet(df, parquet_path)
    return df
