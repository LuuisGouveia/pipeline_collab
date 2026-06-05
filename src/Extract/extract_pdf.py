import os
import re
import json
import time
from google import genai
from google.genai import types
from pypdf import PdfReader
import pandas as pd

client = genai.Client()

def extract_pdf_text(pdf_path):

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
    pattern = r"(\d{2})[\s\-_/](\d{2})[\s\-_/](\d{2,4})"
    match = re.search(pattern, filename)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        return f"{day}/{month}/{year}"
    return None

def structure_data_with_gemini(raw_text):
    
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

    all_services = []

    
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

