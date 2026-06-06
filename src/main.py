from Extract import extract_data
import pandas as pd

if __name__ == "__main__":
    SOURCE_DIR = "./data"  # Pasta de origem contendo PDFs e tabelas tabulares
    df_final = extract_data.extract_data()
    df_final = pd.DataFrame(df_final)  # Garantindo que o resultado seja um DataFrame
    
    if not df_final.empty:
        print("\nDados extraídos com sucesso! Prévia dos dados consolidados:")
        print(df_final.head())
    else:
        print("\n[AVISO] Nenhum dado válido pôde ser extraído dos arquivos.")