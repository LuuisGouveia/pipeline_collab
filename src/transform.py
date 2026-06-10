import os
import pandas as pd
import numpy as np
import re
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
# ==============================================================================
# DICIONÁRIO DE PADRONIZAÇÃO DE TRABALHOS (REGRAS DE NEGÓCIO)
# ==============================================================================
def padronizar_trabalhos(descricao_original):
    """
    Recebe a descrição do trabalho bruto, descarta lançamentos financeiros
    (débitos/valores), avalia as palavras-chave e retorna o termo técnico padronizado.
    """
    if pd.isna(descricao_original):
        return "Não Especificado"

    # Converte para minúsculo para garantir o casamento perfeito das palavras
    texto = str(descricao_original).lower().strip()

    # 🛑 REGRA DE EXCLUSÃO: Se for um registro de débito ou cobrança genérica, marcamos para exclusão
    if re.search(r"d[eé]bito|valor", texto):
        return "EXCLUIR_REGISTRO_FINANCEIRO"

    # Ordem das regras importa: Termos mais específicos vêm antes dos mais gerais!
    regras = [
        # Protocolos e Implantes Específicos
        (
            r"protocolo\s+çer[aâ]mic|protocolo\s+cer[aâ]mic",
            "Protese Protocolo Metalocerâmica",
        ),
        (r"protocolo", "Protese Protocolo Acrilica"),
        (
            r"coroa\s+implante|çer[aâ]mica\s+implante|cer[aâ]mica\s+implante|metalocer[aâ]mica\s+implante",
            "PSI - Coroa Metalocerâmica Sobre Implante",
        ),
        # Dissilicato (E-max e variações)
        (
            r"emax|e-max|dissilicato|litio|l[ií]tio|bloco\s+de\s+emax|laminado|facetas",
            "Coroa/Bloco/Laminado em Dissilicato",
        ),
        # Resinas (Blocos e Coroas Fotopolimerizáveis)
        (
            r"bloco\s+em\s+resina\s+foto|bloco\s+em\s+res\.\s+foto|bloco\s+em\s+resina\s+fotopolimeriz[aá]vel",
            "Bloco em Resina Fotopolimerizável",
        ),
        # 🌟 Atualizado para pegar: Coroa Res Foto, Coroa res. Foto e variações anteriores
        (
            r"coroa\s+em\s+resina\s+foto|coroa\s+em\s+res\.\s+foto|coroa\s+res\.?\s+foto|coroa\s+em\s+resina\s+fotopolimeriz[aá]vel",
            "Coroa em Resina Fotopolimerizável",
        ),
        # Pontes e Próteses Removíveis
        (
            r"perereca|dentadura\s+provis[oó]ria|ppr\s+provis[oó]ria",
            "Protese Parcial Removivel Provisória",
        ),
        (r"ppr|ponte\s+m[oó]vel|estrutura\s+de", "Protese Parcial Removivel"),
        (r"fixa|adesiva", "Ponte Fixa Metalocerâmica"),
        # Próteses Totais e Provisórios Gerais
        (r"dentadura", "Protese Total"),
        # 🌟 Atualizado para pegar: Coroa Res, Coroa Res. e variações anteriores
        (
            r"provis[oó]rio|jaqueta\s+acril[ií]ca|coroa\s+em\s+resina|coroa\s+res\.?$",
            "Provisório",
        ),
        # Coroa Metalocerâmica Geral (Colocado abaixo por conter a palavra "ceramica")
        (
            r"coroa\s+metalocer[aâ]mica|çer[aâ]mica|cer[aâ]mica|metalocer[aâ]mica",
            "Coroa Metalocerâmica",
        ),
    ]

    for padrao, nome_padronizado in regras:
        if re.search(padrao, texto):
            return nome_padronizado

    return str(descricao_original).strip().title()


# ==============================================================================
# 2. CAMADA SILVER (Atualizada com a padronização de serviços)
# ==============================================================================
def _clean_text_column(series):
    """Função interna para padronizar textos gerais (Clientes/Pacientes)"""
    return (
        series.astype(str)
        .str.strip()
        .str.title()
        .replace({"Nan": np.nan, "None": np.nan})
    )


def process_to_silver():
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

        # 🌟 APLICANDO A PADRONIZAÇÃO AQUI
        df_silver["servico_descricao"] = df["servico_trabalho"].apply(
            padronizar_trabalhos
        )

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

        # 🌟 APLICANDO A PADRONIZAÇÃO AQUI
        df_silver["servico_descricao"] = df["servico_trabalho"].apply(
            padronizar_trabalhos
        )

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
        df_silver["data_entrega"] = pd.to_datetime(
            df["Data de Entrega"], dayfirst=True, errors="coerce"
        )
        df_silver["cliente_nome"] = _clean_text_column(df["Cliente (Dentista)"])
        df_silver["paciente_nome"] = _clean_text_column(df["Paciente"])

        # 🌟 APLICANDO A PADRONIZAÇÃO AQUI (Mesmo vindo da IA, garantimos a consistência estrita)
        df_silver["servico_descricao"] = df["Trabalho Executado"].apply(
            padronizar_trabalhos
        )

        df_silver["quantidade"] = (
            pd.to_numeric(df["Quantidade"], errors="coerce").fillna(1).astype(int)
        )
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

    # 🌟 FILTRO DE DATA QUALITY: Elimina os registros de Débito/Valor detectados no mapeamento
    linhas_antes = len(df_silver_final)
    df_silver_final = df_silver_final[
        df_silver_final["servico_descricao"] != "EXCLUIR_REGISTRO_FINANCEIRO"
    ]
    linhas_depois = len(df_silver_final)

    if linhas_antes != linhas_depois:
        print(
            f"[DATA QUALITY] Foram eliminados {linhas_antes - linhas_depois} registros de débitos/valores financeiros."
        )

    # Preenche fallbacks de segurança para nulos após a limpeza
    df_silver_final["cliente_nome"] = df_silver_final["cliente_nome"].fillna(
        "Desconhecido"
    )
    df_silver_final["paciente_nome"] = df_silver_final["paciente_nome"].fillna(
        "Não Especificado"
    )

    # --- REDE DE SEGURANÇA GLOBAL DE DATAS (Mantida do passo anterior) ---
    if df_silver_final["data_entrega"].isna().any():

        def preencher_nat_pelo_nome(row):
            if pd.isna(row["data_entrega"]):
                data_extraida = utils.extract_date_from_filename(row["arquivo_origem"])
                if data_extraida:
                    return pd.to_datetime(data_extraida)
            return row["data_entrega"]

        df_silver_final["data_entrega"] = df_silver_final.apply(
            preencher_nat_pelo_nome, axis=1
        )

    # Adiciona coluna de auditoria do pipeline
    df_silver_final["data_processamento"] = pd.Timestamp.now()

    # Remove duplicados residuais
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
