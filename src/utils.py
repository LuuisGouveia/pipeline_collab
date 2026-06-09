import os
import re
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv


# --- INICIALIZAÇÃO DO .ENV E CONFIGURAÇÃO DO TOKENS ---
def init_environment():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.getcwd()
    possible_paths = [
        os.path.join(current_dir, "config", ".env"),
        os.path.join(current_dir, "..", "config", ".env"),
        os.path.join(current_dir, ".env"),
        os.path.join(working_dir, "config", ".env"),
        os.path.join(working_dir, ".env"),
    ]

    for path in possible_paths:
        normalized_path = os.path.abspath(path)
        if os.path.exists(normalized_path) and load_dotenv(dotenv_path=normalized_path):
            print(f"[DEBUG] .env carregado com sucesso de: {normalized_path}")
            break

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        for env_key, env_value in os.environ.items():
            if "hf" in env_key.lower() and "token" in env_key.lower():
                hf_token = env_value
                print(f"[DEBUG] Chave alternativa de Token HF encontrada: '{env_key}'")
                break
    return hf_token


HF_TOKEN = init_environment()

# --- FUNÇÕES AUXILIARES COMPARTILHADAS ---


def extract_date_from_filename(filename, fallback_carlos=False):
    """Captura Dia, Mês e Ano do nome do arquivo (DD MM YY ou DD MM YYYY)"""
    match = re.search(r"(\d{2})[\s\-_/](\d{2})[\s\-_/](\d{2,4})", filename)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        return f"{year}-{month}-{day}"

    if fallback_carlos and "Carlos" in filename and "2020" in filename:
        return "2020-04-08"

    return None


def clean_monetary_values(v):
    """Sanitiza strings monetárias para float de forma robusta"""
    if pd.isna(v):
        return 0.0
    s = str(v).strip().replace("R$", "").replace("$", "").strip()
    if s.isdigit():
        return float(s)
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0


def save_to_parquet(df, parquet_path):
    """Centraliza a exportação segura para arquivos Parquet"""
    if parquet_path and not df.empty:
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"[SUCESSO] Dados consolidados e salvos em Parquet: {parquet_path}")
        except Exception as e:
            print(f"[ERRO] Falha ao salvar em formato Parquet: {e}")


def structure_data_with_huggingface(raw_text):
    """Envia o texto para a API Hugging Face buscando retorno em JSON"""
    if not HF_TOKEN:
        print("[ERRO] Token do Hugging Face (HF_TOKEN) não configurado!")
        return {"services": []}

    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "És um assistente especialista em extração de dados estruturados em formato JSON.\n"
        "Deves analisar o relatório de prótese dentária enviado e retornar um objeto JSON estrito com o seguinte esquema:\n"
        '{\n  "services": [\n    {\n'
        '      "dentist_client": "Nome do dentista / cliente",\n'
        '      "patient": "Nome do paciente (se não especificado, usa \'Não especificado\')",\n'
        '      "work_performed": "Descrição do trabalho executado",\n'
        '      "unit_value": "Valor unitário formatado como R$ X.XXX,XX",\n'
        '      "quantity": 1,\n'
        '      "total_value": "Valor total formatado como R$ X.XXX,XX",\n'
        '      "delivery_date": "Data de entrega no formato DD/MM/YYYY ou null"\n'
        "    }\n  ]\n}\n"
        "Retorna estritamente apenas o objeto JSON, sem blocos markdown adicionais."
    )

    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Texto do relatório:\n{raw_text}"},
        ],
        "temperature": 0.1,
    }

    wait_times = [3, 6, 12, 24, 48]
    for attempt in range(5):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=45)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                first_brace, last_brace = content.find("{"), content.rfind("}")
                if 0 <= first_brace < last_brace:
                    content = content[first_brace : last_brace + 1]
                return json.loads(content)
            elif response.status_code == 503:
                print(
                    f"[INFO] Modelo carregando no HF. Nova tentativa {attempt + 1}/5..."
                )
            else:
                print(f"[AVISO] Resposta inesperada da API: {response.status_code}")
        except Exception as e:
            if attempt == 4:
                print(f"[ERRO] Falha crítica na API Hugging Face: {e}")
        time.sleep(wait_times[attempt])
    return {"services": []}
