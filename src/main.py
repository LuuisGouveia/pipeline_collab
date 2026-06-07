from Extract import extract_table_lists
from Extract import extract_table_notes
from Extract import extract_pdf
import os
import pandas as pd

if __name__ == "__main__":
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    df_tables = extract_table_lists.extract_table_lists(
        os.path.join(project_root, "pipeline_collab", "data", "lists"),
        os.path.join(
            project_root, "pipeline_collab", "data", "lists_consolidated.parquet"
        ),
    )
    df_notes = extract_table_notes.extract_table_notes(
        os.path.join(project_root, "pipeline_collab", "data", "notes"),
        os.path.join(
            project_root, "pipeline_collab", "data", "notes_consolidated.parquet"
        ),
    )
    df_pdfs = extract_pdf.extract_pdf(
        os.path.join(project_root, "pipeline_collab", "data", "pdf"),
        delay_between_calls=3.0,
        parquet_path=os.path.join(
            project_root, "pipeline_collab", "data", "pdf_consolidated.parquet"
        ),
    )

    # --- CÓDIGO COMPLEMENTAR DE UNIFICAÇÃO (main.py) ---

    # Lista auxiliar para armazenar os blocos que serão concatenados
    dfs_to_concat = []

    # 1. Mapeamento e estruturação do DataFrame de PDFs usando IA
    if df_pdfs is not None and not df_pdfs.empty:
        mapping_pdfs = {
            "Data de Entrega": "delivery_date",
            "Cliente (Dentista)": "client_dentist",
            "Paciente": "patient",
            "Trabalho Executado": "job",
            "Valor Unitário": "price",
            "Quantidade": "quantity",
            "Valor Total": "total_price",
            "Arquivo de Origem": "file_name",
        }
        # Renomeia as colunas e garante apenas as colunas de destino estruturadas
        df_pdfs_mapped = df_pdfs.rename(columns=mapping_pdfs).reindex(
            columns=list(mapping_pdfs.values())
        )
        dfs_to_concat.append(df_pdfs_mapped)

    # 2. Mapeamento e estruturação do DataFrame de Notas Tabulares (table_notes)
    if df_notes is not None and not df_notes.empty:
        mapping_notes = {
            "data_entrega": "delivery_date",
            "cliente_dentista": "client_dentist",
            "paciente": "patient",
            "servico_trabalho": "job",
            "valor_unitario": "price",
            "quantidade": "quantity",
            "valor_total": "total_price",
            "arquivo_origem": "file_name",
        }
        df_notes_mapped = df_notes.rename(columns=mapping_notes).reindex(
            columns=list(mapping_notes.values())
        )
        dfs_to_concat.append(df_notes_mapped)

    # 3. Mapeamento e estruturação do DataFrame de Listas Tabulares (table_lists)
    if df_tables is not None and not df_tables.empty:
        mapping_tables = {
            "data_entrega": "delivery_date",
            "cliente_dentista": "client_dentist",
            "paciente": "patient",
            "servico_trabalho": "job",
            "valor_unitario": "price",
            "quantidade": "quantity",
            "valor_total": "total_price",
            "arquivo_origem": "file_name",
        }
        df_tables_mapped = df_tables.rename(columns=mapping_tables).reindex(
            columns=list(mapping_tables.values())
        )
        dfs_to_concat.append(df_tables_mapped)

    # Concatenação final de todos os dados unificados
    if dfs_to_concat:
        df_unified = pd.concat(dfs_to_concat, ignore_index=True)
        print(
            f"\nDataFrame unificado montado com sucesso! Total de {len(df_unified)} linhas."
        )
        try:
            unified_parquet_path = os.path.join(
                project_root, "pipeline_collab", "data", "unified_data.parquet"
            )
            df_unified.to_parquet(unified_parquet_path, index=False)
            print(
                f"[SUCESSO] DataFrame unificado salvo em Parquet: {unified_parquet_path}"
            )
        except Exception as e:
            print(f"[ERRO] Falha ao salvar DataFrame unificado em Parquet: {e}")
    else:
        # Se nenhum DataFrame tiver dados, retorna uma estrutura limpa e vazia
        df_unified = pd.DataFrame(
            columns=[
                "delivery_date",
                "client_dentist",
                "patient",
                "job",
                "price",
                "quantity",
                "total_price",
                "file_name",
            ]
        )
        print("\n[AVISO] Nenhum dado extraído das fontes para unificação.")

    # O df_unified agora está inteiramente pronto para uso ou persistência (ex: Parquet)
