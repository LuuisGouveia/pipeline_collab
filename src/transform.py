import os
import pandas as pd
import numpy as np
import utils

# Configuração de caminhos para as camadas
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BRONZE_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")

# Garante que as pastas das camadas existam
for folder in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
    os.makedirs(folder, exist_ok=True)


# ==============================================================================
# 1. CAMADA BRONZE (Persistência dos Dados Brutos)
# ==============================================================================
def save_to_bronze(df_notes, df_pdfs, df_lists):
    """
    Salva os dados brutos exatamente como foram extraídos, garantindo que se
    o pipeline falhar adiante, não seja necessário reprocessar os arquivos originais.
    """
    print("\n--- Salvando dados na Camada Bronze ---")
    if df_notes is not None and not df_notes.empty:
        utils.save_to_parquet(
            df_notes, os.path.join(BRONZE_DIR, "raw_notes_csv.parquet")
        )
    if df_pdfs is not None and not df_pdfs.empty:
        utils.save_to_parquet(df_pdfs, os.path.join(BRONZE_DIR, "raw_pdfs_ia.parquet"))
    if df_lists is not None and not df_lists.empty:
        utils.save_to_parquet(
            df_lists, os.path.join(BRONZE_DIR, "raw_lists_csv.parquet")
        )


# ==============================================================================
# 2. CAMADA SILVER (Padronização, Limpeza e Enriquecimento)
# ==============================================================================
def _clean_text_column(series):
    """Função interna para padronizar textos (Title Case, sem espaços extras)"""
    return (
        series.astype(str)
        .str.strip()
        .str.title()
        .replace({"Nan": np.nan, "None": np.nan})
    )


def process_to_silver():
    """
    Lê os dados da Bronze, unifica os schemas de fontes diferentes,
    limpa strings e garante a tipagem correta de cada coluna.
    """
    print("\n--- Transformando dados para a Camada Silver ---")

    silver_dfs = []

    # 2.1 Processando Notas CSV
    path_notes = os.path.join(BRONZE_DIR, "raw_notes_csv.parquet")
    if os.path.exists(path_notes):
        df = pd.read_parquet(path_notes)
        df_silver = pd.DataFrame()
        df_silver["data_entrega"] = pd.to_datetime(df["data_entrega"], errors="coerce")
        df_silver["cliente_nome"] = _clean_text_column(df["cliente_dentista"])
        df_silver["paciente_nome"] = _clean_text_column(df["paciente"])
        df_silver["servico_descricao"] = _clean_text_column(df["servico_trabalho"])
        df_silver["quantidade"] = (
            pd.to_numeric(df["quantidade"], errors="coerce").fillna(1).astype(int)
        )
        df_silver["valor_unitario"] = pd.to_numeric(
            df["valor_unitario"], errors="coerce"
        ).fillna(0.0)
        df_silver["valor_total"] = pd.to_numeric(
            df["valor_total"], errors="coerce"
        ).fillna(0.0)
        df_silver["arquivo_origem"] = df["arquivo_origem"]
        df_silver["tipo_fonte"] = "Notas CSV"
        silver_dfs.append(df_silver)

    # 2.2 Processando Listas CSV
    path_lists = os.path.join(BRONZE_DIR, "raw_lists_csv.parquet")
    if os.path.exists(path_lists):
        df = pd.read_parquet(path_lists)
        df_silver = pd.DataFrame()
        df_silver["data_entrega"] = pd.to_datetime(df["data_entrega"], errors="coerce")
        df_silver["cliente_nome"] = _clean_text_column(df["cliente_dentista"])
        df_silver["paciente_nome"] = _clean_text_column(df["paciente"])
        df_silver["servico_descricao"] = _clean_text_column(df["servico_trabalho"])
        df_silver["quantidade"] = (
            pd.to_numeric(df["quantidade"], errors="coerce").fillna(1).astype(int)
        )
        df_silver["valor_unitario"] = pd.to_numeric(
            df["valor_unitario"], errors="coerce"
        ).fillna(0.0)
        df_silver["valor_total"] = pd.to_numeric(
            df["valor_total"], errors="coerce"
        ).fillna(0.0)
        df_silver["arquivo_origem"] = df["arquivo_origem"]
        df_silver["tipo_fonte"] = "Listas CSV"
        silver_dfs.append(df_silver)

    # 2.3 Processando PDFs (IA)
    path_pdfs = os.path.join(BRONZE_DIR, "raw_pdfs_ia.parquet")
    if os.path.exists(path_pdfs):
        df = pd.read_parquet(path_pdfs)
        df_silver = pd.DataFrame()
        # Tratamento da data vinda da IA (converte formato DD/MM/YYYY se necessário)
        df_silver["data_entrega"] = pd.to_datetime(
            df["Data de Entrega"], dayfirst=True, errors="coerce"
        )
        df_silver["cliente_nome"] = _clean_text_column(df["Cliente (Dentista)"])
        df_silver["paciente_nome"] = _clean_text_column(df["Paciente"])
        df_silver["servico_descricao"] = _clean_text_column(df["Trabalho Executado"])
        df_silver["quantidade"] = (
            pd.to_numeric(df["Quantidade"], errors="coerce").fillna(1).astype(int)
        )

        # Como os valores da IA vêm formatados com 'R$', limpamos usando a utilidade existente
        df_silver["valor_unitario"] = df["Valor Unitário"].apply(
            utils.clean_monetary_values
        )
        df_silver["valor_total"] = df["Valor Total"].apply(utils.clean_monetary_values)

        df_silver["arquivo_origem"] = df["Arquivo de Origem"]
        df_silver["tipo_fonte"] = "PDF Inteligente"
        silver_dfs.append(df_silver)

    if not silver_dfs:
        print("[AVISO] Nenhum dado encontrado na camada Bronze para processar.")
        return pd.DataFrame()

    # Consolida as 3 fontes em uma única tabela Silver padronizada
    df_silver_final = pd.concat(silver_dfs, ignore_index=True)

    # --- 🛡️ REDE DE SEGURANÇA GLOBAL (DATA QUALITY) 🛡️ ---
    # Se qualquer registro de qualquer fonte ficou com data NaT, tenta pescar a data pelo nome do arquivo
    if df_silver_final["data_entrega"].isna().any():
        print(
            "[DATA QUALITY] Identificadas datas nulas (NaT). Aplicando correção via nome do arquivo de origem..."
        )

        def preencher_nat_pelo_nome(row):
            if pd.isna(row["data_entrega"]):
                data_extraida = utils.extract_date_from_filename(row["arquivo_origem"])
                if data_extraida:
                    return pd.to_datetime(data_extraida)
            return row["data_entrega"]

        df_silver_final["data_entrega"] = df_silver_final.apply(
            preencher_nat_pelo_nome, axis=1
        )

    # Consolida as 3 fontes em uma única tabela Silver padronizada
    df_silver_final = pd.concat(silver_dfs, ignore_index=True)

    # Preenche fallbacks de segurança para nulos após a limpeza
    df_silver_final["cliente_nome"] = df_silver_final["cliente_nome"].fillna(
        "Desconhecido"
    )
    df_silver_final["paciente_nome"] = df_silver_final["paciente_nome"].fillna(
        "Não Especificado"
    )
    df_silver_final["servico_descricao"] = df_silver_final["servico_descricao"].fillna(
        "Não Especificado"
    )

    # Adiciona coluna de auditoria do pipeline
    df_silver_final["data_processamento"] = pd.Timestamp.now()

    # Remove duplicados que possam ter passado entre as consolidações
    df_silver_final.drop_duplicates(
        subset=[
            "data_entrega",
            "cliente_nome",
            "paciente_nome",
            "servico_descricao",
            "valor_total",
        ],
        inplace=True,
    )
    df_silver_final.reset_index(drop=True, inplace=True)

    # Salva o resultado final na Silver
    utils.save_to_parquet(
        df_silver_final, os.path.join(SILVER_DIR, "cleansed_services.parquet")
    )
    print(
        f"[SUCESSO] Camada Silver gerada com {len(df_silver_final)} registros padronizados."
    )
    return df_silver_final


# ==============================================================================
# 3. CAMADA GOLD (Modelagem Analítica / Fatos e Dimensões)
# ==============================================================================
def generate_gold_layer(df_silver):
    """
    Quebra a tabela Silver unificada em um modelo de tabelas Fato e Dimensões (Star Schema),
    perfeito para o carregamento em bancos relacionais ou BI.
    """
    print("\n--- Modelando dados para a Camada Gold ---")
    if df_silver is not None and df_silver.empty:
        return

    # 3.1 DIMENSÃO: Clientes (Dentistas)
    # Coleta nomes únicos e gera um ID incremental estável baseado no nome ordenado
    unique_clients = sorted(df_silver["cliente_nome"].unique())
    df_dim_clientes = pd.DataFrame({"cliente_nome": unique_clients})
    df_dim_clientes.insert(
        0, "sk_cliente", range(1, len(df_dim_clientes) + 1)
    )  # surrogate key

    # 3.2 DIMENSÃO: Serviços / Trabalhos
    unique_services = sorted(df_silver["servico_descricao"].unique())
    df_dim_servicos = pd.DataFrame({"servico_descricao": unique_services})
    df_dim_servicos.insert(0, "sk_servico", range(1, len(df_dim_servicos) + 1))

    # 3.3 TABELA FATO: Faturamento
    # Mapeia os IDs das dimensões de volta para a tabela principal para criar as chaves estrangeiras
    df_fato = df_silver.merge(df_dim_clientes, on="cliente_nome", how="left")
    df_fato = df_fato.merge(df_dim_servicos, on="servico_descricao", how="left")

    # Seleciona e ordena as colunas focando em métricas de negócio e chaves analíticas
    columns_fato = [
        "data_entrega",
        "sk_cliente",
        "sk_servico",
        "paciente_nome",  # Mantido na fato por ser de altíssima cardinalidade e granularidade
        "quantidade",
        "valor_unitario",
        "valor_total",
        "tipo_fonte",
        "arquivo_origem",
    ]
    df_fato = df_fato[columns_fato]
    df_fato.insert(0, "id_fato", range(1, len(df_fato) + 1))

    # Salvando as tabelas finais da Gold
    utils.save_to_parquet(
        df_dim_clientes, os.path.join(GOLD_DIR, "dim_clientes.parquet")
    )
    utils.save_to_parquet(
        df_dim_servicos, os.path.join(GOLD_DIR, "dim_servicos.parquet")
    )
    utils.save_to_parquet(df_fato, os.path.join(GOLD_DIR, "fato_faturamento.parquet"))

    print(f"[SUCESSO] Camada Gold concluída!")
    print(f"          -> fat_faturamento: {len(df_fato)} linhas")
    print(f"          -> dim_clientes: {len(df_dim_clientes)} registros")
    print(f"          -> dim_servicos: {len(df_dim_servicos)} registros")
