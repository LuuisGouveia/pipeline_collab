import os
import processors
import transform  # Novo import do nosso script de transformação

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    notes_csv_path = os.path.join(project_root, "data", "notes")
    lists_csv_path = os.path.join(project_root, "data", "lists")
    pdf_path = os.path.join(project_root, "data", "pdf")
    report_csv_path = os.path.join(project_root, "data", "reports")

    # === 1. ETAPA DE EXTRAÇÃO (E CRIAÇÃO DA BRONZE) ===
    print("\n[PIPELINE] Iniciando Fase de Extração...")
    df_notes = processors.batch_process_folder(notes_csv_path, process_type="note_csv")
    df_pdfs = processors.batch_process_folder(pdf_path, process_type="pdf", delay=3.0)
    df_lists = processors.batch_process_folder(lists_csv_path, process_type="list_csv")
    df_reports = processors.batch_process_folder(
        report_csv_path, process_type="report_csv"
    )

    # Mover os dados extraídos bruto diretamente para a Bronze
    transform.save_to_bronze(df_notes, df_pdfs, df_lists, df_reports)

    # === 2. ETAPA DE TRANSFORMAÇÃO (SILVER) ===
    # Lê os dados salvos na Bronze, limpa, unifica o schema e tipa as colunas
    df_silver = transform.process_to_silver()

    # === 3. ETAPA DE MODELAGEM ANALÍTICA (GOLD) ===
    # Cria o modelo dimensional (Fato/Dimensões) pronto para o Load
    transform.generate_gold_layer(df_silver)

    print("\n[PIPELINE] Executado com sucesso da Extração até a Camada Gold!")
