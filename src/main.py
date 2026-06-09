import os
from Extract.processors import batch_process_folder

if __name__ == "__main__":
    # Define a raiz dinâmica do projeto
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Caminhos de entrada
    notes_csv_path = os.path.join(project_root, "data", "notes")
    lists_csv_path = os.path.join(project_root, "data", "lists")
    pdf_path = os.path.join(project_root, "data", "pdf")

    # 1. Executando o Bloco 1 (Notas CSV)
    print("\n--- INICIANDO PROCESSAMENTO DE NOTAS CSV ---")
    df_notes = batch_process_folder(notes_csv_path, process_type="note_csv")
    if not df_notes.empty:
        print(df_notes.head())

    # 2. Executando o Bloco 2 (PDFs via IA)
    print("\n--- INICIANDO PROCESSAMENTO DE RELATÓRIOS PDF ---")
    df_pdfs = batch_process_folder(pdf_path, process_type="pdf", delay=3.0)
    if not df_pdfs.empty:
        print(df_pdfs.head())

    # 3. Executando o Bloco 3 (Listas CSV)
    print("\n--- INICIANDO PROCESSAMENTO DE LISTAS CSV ---")
    df_lists = batch_process_folder(lists_csv_path, process_type="list_csv")
    if not df_lists.empty:
        print(df_lists.head())
