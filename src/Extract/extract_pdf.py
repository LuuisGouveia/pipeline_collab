import os
import re
import json
import time
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
import pandas as pd
import numpy as np

# --- BLOCO DE INICIALIZAÇÃO ULTRA-ROBUSTO DO ARQUIVO .ENV ---
current_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.getcwd()

# Lista de caminhos possíveis para o ficheiro .env
possible_env_paths = [
    os.path.join(
        current_dir, "config", ".env"
    ),  # Se o script estiver na raiz e o .env em raiz/config/.env
    os.path.join(
        current_dir, "..", "config", ".env"
    ),  # Se o script estiver em raiz/src e o .env em raiz/config/.env
    os.path.join(current_dir, ".env"),  # Se o .env estiver na mesma pasta do script
    os.path.join(
        working_dir, "config", ".env"
    ),  # Com base no diretório ativo do terminal
    os.path.join(
        working_dir, ".env"
    ),  # Se o .env estiver no diretório atual do terminal
]

env_loaded = False
loaded_path = None

for path in possible_env_paths:
    normalized_path = os.path.abspath(path)
    if os.path.exists(normalized_path):
        env_loaded = load_dotenv(dotenv_path=normalized_path)
        if env_loaded:
            loaded_path = normalized_path
            break

if env_loaded:
    print(f"[DEBUG] Ficheiro .env localizado e carregado com sucesso de: {loaded_path}")
else:
    print("[AVISO] Não foi possível encontrar o ficheiro .env nos caminhos testados.")
    print("[DEBUG] Caminhos verificados:")
    for path in possible_env_paths:
        print(f"  - {os.path.abspath(path)}")

# Resgata o Token do Hugging Face das variáveis de ambiente carregadas
hf_token = os.getenv("HF_TOKEN")

# Fallback inteligente caso a variável tenha sido escrita com variações de letras maiúsculas/minúsculas
if not hf_token:
    for env_key, env_value in os.environ.items():
        if "hf" in env_key.lower() and "token" in env_key.lower():
            hf_token = env_value
            print(f"[DEBUG] Chave alternativa de Token HF encontrada: '{env_key}'")
            break

if hf_token:
    masked_key = hf_token[:6] + "..." + hf_token[-4:] if len(hf_token) > 10 else "***"
    print(f"[DEBUG] Token do Hugging Face carregado com sucesso: {masked_key}")
else:
    print("[ERRO CRÍTICO] A variável de ambiente 'HF_TOKEN' não foi encontrada!")
    print(
        "[DICA] Certifica-te de que o teu ficheiro .env possui a linha: HF_TOKEN=o_teu_token_aqui"
    )


def extract_pdf_text(pdf_path):
    """Extrai todo o texto legível de um ficheiro PDF."""
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


def extract_date_from_filename(filename):
    """
    Tenta extrair uma data do nome do ficheiro no formato DD MM YY ou DD MM YYYY.
    Exemplo: 'CARLOS 01 10 20.pdf' -> '01/10/2020'
    Exemplo: 'Aline Vasconcelos 31 08 2023.pdf' -> '31/08/2023'
    """
    pattern = r"(\d{2})[\s\-_/](\d{2})[\s\-_/](\d{2,4})"
    match = re.search(pattern, filename)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        return f"{day}/{month}/{year}"
    return None


def structure_data_with_huggingface(raw_text):
    """
    Envia o texto bruto para um modelo gratuito do Hugging Face (via API de Inferência)
    e solicita um retorno estruturado em JSON com as chaves definidas.
    Implementa retentativas com backoff exponencial para lidar com tempos de carregamento do modelo.
    """
    if not hf_token:
        print("[ERRO] Token do Hugging Face (HF_TOKEN) não configurado!")
        return {"services": []}

    # Utilizamos o Qwen 2.5 72B Instruct, que é excelente para JSON e português
    model_id = "Qwen/Qwen2.5-72B-Instruct"
    api_url = f"https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    # Prompt de sistema definindo o esquema esperado de forma explícita
    system_prompt = (
        "És um assistente especialista em extração de dados estruturados em formato JSON.\n"
        "Deves analisar o relatório de prótese dentária enviado e retornar um objeto JSON estrito com o seguinte esquema:\n"
        "{\n"
        '  "services": [\n'
        "    {\n"
        '      "dentist_client": "Nome do dentista / cliente",\n'
        '      "patient": "Nome do paciente (se não especificado, usa \'Não especificado\')",\n'
        '      "work_performed": "Descrição do trabalho executado",\n'
        '      "unit_value": "Valor unitário formatado como R$ X.XXX,XX (se não explícito, divide o valor total pela quantidade)",\n'
        '      "quantity": 1,\n'
        '      "total_value": "Valor total formatado como R$ X.XXX,XX",\n'
        '      "delivery_date": "Data de entrega no formato DD/MM/YYYY ou null se não houver na tabela"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Retorna estritamente apenas o objeto JSON. Não adiciones blocos de código markdown adicionais ou textos introdutórios/explicativos."
    )

    # Removido 'response_format' para evitar o erro HTTP 400 em servidores serverless compartilhados
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Texto do relatório:\n{raw_text}"},
        ],
        "temperature": 0.1,
    }

    max_retries = 5
    wait_times = [3, 6, 12, 24, 48]

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=45)

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()

                # Resiliência: Extrai apenas a string contida entre a primeira chave '{' e a última '}'
                # Isso impede falhas caso o modelo decida retornar markdown como "```json ... ```" ou tags adicionais
                first_brace = content.find("{")
                last_brace = content.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    content = content[first_brace : last_brace + 1]

                return json.loads(content)

            elif response.status_code == 503:
                print(
                    f"[INFO] O modelo {model_id} está a carregar no Hugging Face. Nova tentativa {attempt + 1}/{max_retries}..."
                )

            else:
                print(
                    f"[AVISO] Resposta inesperada da API Hugging Face: {response.status_code}. Detalhes: {response.text}"
                )

        except Exception as e:
            if attempt == max_retries - 1:
                print(
                    f"\n[ERRO] Falha crítica ao comunicar com o Hugging Face após {max_retries} tentativas: {e}"
                )
                return {"services": []}

        time.sleep(wait_times[attempt])

    return {"services": []}


def extract_pdf(source_dir, delay_between_calls=3.0, parquet_path=None):
    """
    Varre a pasta de PDFs de origem, extrai os dados usando Inteligência Artificial,
    aplica regras de tratamento síncronas, opcionalmente exporta para Parquet e retorna um Pandas DataFrame.
    """
    all_services = []

    if not os.path.exists(source_dir):
        print(f"Aviso: A pasta de origem '{source_dir}' não existe.")
        return pd.DataFrame()

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]

    if not files:
        print(f"Aviso: Nenhum ficheiro PDF encontrado na pasta '{source_dir}'.")
        return pd.DataFrame()

    total_files = len(files)
    print(f"Iniciando o processamento de {total_files} ficheiros PDF...")

    for idx, filename in enumerate(files, start=1):
        full_path = os.path.join(source_dir, filename)
        print(f"[{idx}/{total_files}] Processando: {filename}")

        filename_date_fallback = extract_date_from_filename(filename)
        raw_text = extract_pdf_text(full_path)

        if not raw_text.strip():
            print(f"Aviso: O ficheiro '{filename}' está sem texto ou corrompido.")
            continue

        json_data = structure_data_with_huggingface(raw_text)

        for service in json_data.get("services", []):
            final_date = service.get("delivery_date")

            if not final_date or final_date.strip() == "":
                final_date = (
                    filename_date_fallback
                    if filename_date_fallback
                    else "Não especificada"
                )

            service["final_delivery_date"] = final_date
            service["source_file"] = filename
            all_services.append(service)

        if idx < total_files and delay_between_calls > 0:
            time.sleep(delay_between_calls)

    if all_services:
        df = pd.DataFrame(all_services)

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

        if parquet_path:
            try:
                df.to_parquet(parquet_path, index=False)
                print(
                    f"[SUCESSO] Dados consolidados e salvos em formato Parquet: {parquet_path}"
                )
            except Exception as e:
                print(f"[ERRO] Falha ao salvar em formato Parquet: {e}")

        print(f"\nProcessamento concluído com sucesso! {len(df)} registos extraídos.")
        return df
    else:
        print("\nAviso: Nenhum dado foi encontrado nos PDFs processados.")
        return pd.DataFrame()


if __name__ == "__main__":
    # Exemplo de execução direta do script para teste
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    pdf_path = os.path.join(project_root, "data", "pdf")
    df_extracted = extract_pdf(pdf_path, delay_between_calls=3.0)
    print(df_extracted.head())
