import os
import re
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
import pandas as pd

# --- BLOCO DE INICIALIZAÇÃO ULTRA-ROBUSTO DO ARQUIVO .ENV ---
current_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.getcwd()

# Lista de caminhos possíveis para o arquivo .env
possible_env_paths = [
    os.path.join(current_dir, 'config', '.env'),         # Se o script estiver na raiz e o .env em raiz/config/.env
    os.path.join(current_dir, '..', 'config', '.env'),    # Se o script estiver em raiz/src e o .env em raiz/config/.env
    os.path.join(current_dir, '.env'),                    # Se o .env estiver na mesma pasta do script
    os.path.join(working_dir, 'config', '.env'),         # Com base no diretório ativo do terminal
    os.path.join(working_dir, '.env'),                    # Se o .env estiver no diretório atual do terminal
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
    print(f"[DEBUG] Arquivo .env localizado e carregado com sucesso de: {loaded_path}")
else:
    print("[AVISO] Não foi possível encontrar o arquivo .env nos caminhos testados.")
    print("[DEBUG] Caminhos verificados:")
    for path in possible_env_paths:
        print(f"  - {os.path.abspath(path)}")

# Resgata a API Key das variáveis de ambiente carregadas
api_key = os.getenv("GEMINI_API_KEY")

# Fallback inteligente caso a variável tenha sido escrita com variações de letras maiúsculas/minúsculas
if not api_key:
    for env_key, env_value in os.environ.items():
        if 'gemini' in env_key.lower() and 'key' in env_key.lower():
            api_key = env_value
            print(f"[DEBUG] Chave alternativa encontrada: '{env_key}'")
            break

if api_key:
    # Exibe apenas os caracteres externos da chave no terminal por segurança
    masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
    print(f"[DEBUG] Chave API carregada com sucesso: {masked_key}")
else:
    print("[ERRO CRÍTICO] A variável de ambiente 'GEMINI_API_KEY' não foi encontrada!")
    print("[DICA] Certifique-se de que seu arquivo .env possui a linha: GEMINI_API_KEY=sua_chave_aqui")

# Inicializa o cliente fornecendo explicitamente a chave
# Isso contorna problemas de escopo em que a biblioteca falha ao tentar buscar variáveis globais
client = genai.Client(api_key=api_key)


def extract_pdf_text(pdf_path):
    """Extrai todo o texto legível de um arquivo PDF."""
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
    Tenta extrair uma data do nome do arquivo no formato DD MM YY ou DD MM YYYY.
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


def structure_data_with_gemini(raw_text):
    """
    Envia o texto bruto ao Gemini e solicita retorno estritamente estruturado em JSON.
    Implementa retentativas com backoff exponencial para lidar de forma robusta com limites de taxa.
    """
    output_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "services": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "dentist_client": types.Schema(type=types.Type.STRING),
                        "patient": types.Schema(type=types.Type.STRING),
                        "work_performed": types.Schema(type=types.Type.STRING),
                        "unit_value": types.Schema(type=types.Type.STRING),
                        "quantity": types.Schema(type=types.Type.INTEGER),
                        "total_value": types.Schema(type=types.Type.STRING),
                        "delivery_date": types.Schema(type=types.Type.STRING, nullable=True) 
                    },
                    required=["dentist_client", "patient", "work_performed", "unit_value", "quantity", "total_value"]
                )
            )
        },
        required=["services"]
    )

    prompt = (
        "Você é um assistente especialista em extração de dados de relatórios de prótese dentária.\n"
        "Analise o texto bruto do PDF e extraia cada serviço listado para o formato JSON esperado.\n"
        "Mapeie as informações para as seguintes chaves em inglês:\n"
        "- 'dentist_client': Nome do dentista / cliente.\n"
        "- 'patient': Nome do paciente (se não especificado, use 'Não especificado').\n"
        "- 'work_performed': Descrição do trabalho executado.\n"
        "- 'unit_value': Valor unitário. Se não estiver explícito, divida o valor total pela quantidade para encontrá-lo.\n"
        "- 'quantity': Quantidade (número inteiro).\n"
        "- 'total_value': Valor total do item.\n"
        "- 'delivery_date': Data de entrega (sob a coluna 'Dt Entregue', 'Dt Entrada Dt Entregue Qtd' ou similares).\n"
        "  Se houver data individual na tabela, coloque-a no formato DD/MM/YYYY. Se não houver, deixe como null.\n\n"
        "Regras de formatação:\n"
        "- Formate todos os valores monetários como 'R$ X.XXX,XX'."
    )

    max_retries = 5
    wait_times = [1, 2, 4, 8, 16]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, raw_text],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=output_schema,
                    temperature=0.1
                ),
            )
            return json.loads(response.text)
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[ERRO] Não foi possível estruturar os dados deste PDF após {max_retries} tentativas. Detalhes: {e}")
                return {"services": []}
            
            time.sleep(wait_times[attempt])


def extract_pdf(source_dir, delay_between_calls=3.0):
    """
    Varre a pasta de PDFs de origem, extrai os dados usando Inteligência Artificial,
    aplica regras de tratamento síncronas e retorna um Pandas DataFrame unificado.
    """
    all_services = []

    if not os.path.exists(source_dir):
        print(f"Aviso: A pasta de origem '{source_dir}' não existe.")
        return pd.DataFrame()

    files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]
    
    if not files:
        print(f"Aviso: Nenhum arquivo PDF encontrado na pasta '{source_dir}'.")
        return pd.DataFrame()

    total_files = len(files)
    print(f"Iniciando o processamento de {total_files} arquivos PDF...")

    for idx, filename in enumerate(files, start=1):
        full_path = os.path.join(source_dir, filename)
        print(f"[{idx}/{total_files}] Processando: {filename}")
        
        filename_date_fallback = extract_date_from_filename(filename)
        raw_text = extract_pdf_text(full_path)
        
        if not raw_text.strip():
            print(f"Aviso: O arquivo '{filename}' está sem texto ou corrompido.")
            continue

        json_data = structure_data_with_gemini(raw_text)

        for service in json_data.get("services", []):
            final_date = service.get("delivery_date")

            if not final_date or final_date.strip() == "":
                final_date = filename_date_fallback if filename_date_fallback else "Não especificada"
                
            service["final_delivery_date"] = final_date
            service["source_file"] = filename
            all_services.append(service)

        if idx < total_files and delay_between_calls > 0:
            time.sleep(delay_between_calls)

    if all_services:
        df = pd.DataFrame(all_services)

        ordered_columns = [
            "dentist_client", "patient", "work_performed", 
            "unit_value", "quantity", "total_value", "final_delivery_date"
        ]
        df = df[ordered_columns]
        df.columns = [
            "Cliente (Dentista)", "Paciente", "Trabalho Executado", 
            "Valor Unitário", "Quantidade", "Valor Total", "Data de Entrega"
        ]
        
        print(f"\nProcessamento concluído com sucesso! {len(df)} registros extraídos.")
        return df
    else:
        print("\nAviso: Nenhum dado foi encontrado nos PDFs processados.")
        return pd.DataFrame()


# Exemplo de uso local
if __name__ == "__main__":
    # Ajuste o caminho da pasta de entrada conforme seu ambiente
    INPUT_DIR = "./data/pdf"  # Pasta contendo os arquivos PDF a serem processados
    
    # Executa a extração
    df_result = extract_pdf(INPUT_DIR, delay_between_calls=3.0)
    
    if not df_result.empty:
        print("\nVisualização das primeiras linhas extraídas:")
        print(df_result.head())