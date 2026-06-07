from extract_table_lists import extract_table_lists
from extract_table_notes import extract_table_notes
from extract_pdf import extract_pdf
import pandas as pd
import os


def extract_data():
    """
    Função principal para extrair dados de notas de serviço odontológico.
    Varre a pasta de origem, processa arquivos PDF e tabulares, e consolida tudo em um DataFrame unificado.
    """

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    lists_path = os.path.join(
        project_root, "data", "lists"
    )  # Pasta contendo arquivos tabulares de listas
    notes_path = os.path.join(
        project_root, "data", "notes"
    )  # Pasta contendo arquivos tabulares de notas
    pdf_path = os.path.join(
        project_root, "data", "pdf"
    )  # Pasta contendo arquivos PDF a serem processados

    # Extrai dados de tabelas tabulares (CSV/tabular)
    df_tables = extract_table_lists(lists_path)

    # Extrai dados de tabelas de notas (CSV/tabular)
    df_notes = extract_table_notes(notes_path)

    # Extrai dados de arquivos PDF usando IA
    df_pdfs = extract_pdf(pdf_path, delay_between_calls=3.0)

    # Consolida todos os DataFrames em um único DataFrame unificado
    all_dfs = [
        df for df in [df_tables, df_notes, df_pdfs] if df is not None and not df.empty
    ]

    if all_dfs:
        df_consolidated = pd.concat(all_dfs, ignore_index=True)
        print(f"Dados consolidados: {df_consolidated.shape[0]} registros extraídos.")
        return df_consolidated
    else:
        print("\n[AVISO] Nenhum dado válido pôde ser extraído dos arquivos.")
        return pd.DataFrame()
