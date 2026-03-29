import pandas as pd
from groq import Groq
import time
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import base64

# --- CONFIGURAÇÃO ---
# Lembrete: Revogue esta chave no painel do Groq assim que terminar o projeto!
GROQ_API_KEY = "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"
# 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
# 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
client = Groq(api_key=GROQ_API_KEY)

# Caminho dos arquivos (mantido o mesmo para ler o cache das 350 chamadas já feitas)
input_csv = "input_image_llm.csv"
output_csv = "classificacao_image_llama90b.csv"

# VARIÁVEIS DE CONTROLE
MAX_CALLS_PER_RUN = 416
api_calls_made_this_run = 0

def encode_image(image_path):
    """Lê a imagem e converte para texto base64 para enviar via API."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Erro ao ler imagem {image_path}: {e}")
        return None

# 1. CARREGAR RESULTADOS ANTERIORES (CACHE)
existing_classifications = {}
if os.path.exists(output_csv):
    try:
        print(f"Carregando resultados anteriores de '{output_csv}'...")
        df_output_existing = pd.read_csv(output_csv)
        
        valid_classifications = df_output_existing[
            pd.notna(df_output_existing['Classificacao_LLM']) &
            ~df_output_existing['Classificacao_LLM'].str.startswith('Erro:') &
            (df_output_existing['Classificacao_LLM'] != 'Pending')
        ]
        
        existing_classifications = pd.Series(
            valid_classifications.Classificacao_LLM.values,
            index=valid_classifications.ID
        ).to_dict()
        
    except (pd.errors.EmptyDataError, KeyError):
        print(f"Arquivo '{output_csv}' encontrado, vazio ou corrompido. Começando do zero.")
    except Exception as e:
        print(f"Erro ao ler cache '{output_csv}': {e}. Começando do zero.")
else:
    print(f"Arquivo de saída '{output_csv}' não encontrado. Começando do zero.")

total_ja_processados = len(existing_classifications)

# 2. LÊ O ARQUIVO DE ENTRADA PRINCIPAL
try:
    df_input = pd.read_csv(input_csv)
except FileNotFoundError:
    print(f"Erro: Arquivo de entrada '{input_csv}' não encontrado.")
    sys.exit()

total_a_processar = len(df_input) - total_ja_processados

print(f"\n--- Status do Processamento ---")
print(f"Total de Imagens: {len(df_input)}")
print(f"Já Processados (Cache): {total_ja_processados}")
print(f"Restantes a Processar: {total_a_processar}")
print(f"Limite de NOVAS chamadas para esta execução: {MAX_CALLS_PER_RUN}\n")

resultados = []

# 3. LOOP PRINCIPAL
for i, row in df_input.iterrows():
    image_id = row["ID"]
    file_path = row["Image_path"]
    
    # 1. Já foi classificado
    if image_id in existing_classifications:
        classificacao = existing_classifications[image_id]
        resultados.append(classificacao)
    
    # 2. Limite atingido
    elif api_calls_made_this_run >= MAX_CALLS_PER_RUN:
        resultados.append("Pending")
        # Print comentado para não poluir a tela já que temos 350 no cache
        # print(f"({i+1}/{len(df_input)}) ID={image_id}: Limite estipulado ({MAX_CALLS_PER_RUN}) atingido.")
    
    # 3. Chamar API do Groq
    else:
        print(f"Classificando imagem {i+1}/{len(df_input)}: ID={image_id} (Chamada {api_calls_made_this_run + 1}/{MAX_CALLS_PER_RUN})")
        
        base64_image = encode_image(file_path)
        
        if not base64_image:
            resultados.append("Erro: Falha ao converter imagem")
            continue

        try:
            prompt_text = (
                "You are an expert model in detecting diabetic retinopathy.\n"
                "Classify the attached medical image into one of the following categories:\n"
                "- Healthy\n"
                "- Mild DR\n"
                "- Moderate DR\n"
                "- Proliferative DR\n"
                "- Severe DR\n\n"
                "Answer ONLY with the exact name of the category. Do not add any explanation."
            )

            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct", 
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1, 
                max_completion_tokens=10
            )

            classificacao = response.choices[0].message.content.strip()
            classificacao = classificacao.replace(".", "").replace("**", "")

            print(f"Resultado: {classificacao}\n")
            resultados.append(classificacao)
            api_calls_made_this_run += 1
            
            time.sleep(5)

        except Exception as e:
            print(f"Erro ao processar ID={image_id}: {str(e)}\n")
            resultados.append(f"Erro: {str(e)}")
            api_calls_made_this_run += 1
            time.sleep(10) 

# 4. SALVAR RESULTADOS
df_input["Classificacao_LLM"] = resultados

print(f"\n{api_calls_made_this_run} novas chamadas de API realizadas nesta execução.")
print(f"Total de chamadas acumuladas (salvas): {total_ja_processados + api_calls_made_this_run}")
df_input.to_csv(output_csv, index=False)
print(f"Resultados salvos em {output_csv}")

# 5. GERAR MÉTRICAS (CORRIGIDO PARA IGNORAR CASE SENSITIVE)
print("\nGerando métricas de avaliação...")

TRUE_LABEL_COLUMN_NAME = 'Label_Verdadeiro'

if TRUE_LABEL_COLUMN_NAME not in df_input.columns:
    print(f"Erro: A coluna '{TRUE_LABEL_COLUMN_NAME}' não foi encontrada no CSV.")
else:
    labels_order = ["Healthy", "Mild DR", "Moderate DR", "Proliferative DR", "Severe DR"]
    
    df_valid = df_input[
        pd.notna(df_input['Classificacao_LLM']) &
        ~df_input['Classificacao_LLM'].str.startswith('Erro:') &
        (df_input['Classificacao_LLM'] != 'Pending')
    ].copy()
    
    if len(df_valid) == 0:
        print("Nenhuma amostra válida encontrada para gerar métricas.")
    else:
        # Dicionário à prova de balas
        mapeamento = {
            "healthy": "Healthy", "0": "Healthy", "0.0": "Healthy",
            "mild dr": "Mild DR", "1": "Mild DR", "1.0": "Mild DR",
            "moderate dr": "Moderate DR", "2": "Moderate DR", "2.0": "Moderate DR",
            "proliferative dr": "Proliferative DR", "3": "Proliferative DR", "3.0": "Proliferative DR",
            "severe dr": "Severe DR", "4": "Severe DR", "4.0": "Severe DR"
        }
        
        # Padronização e limpeza
        df_valid[TRUE_LABEL_COLUMN_NAME] = df_valid[TRUE_LABEL_COLUMN_NAME].astype(str).str.strip().str.lower()
        df_valid[TRUE_LABEL_COLUMN_NAME] = df_valid[TRUE_LABEL_COLUMN_NAME].map(lambda x: mapeamento.get(x, "Desconhecido"))
        
        df_valid['Classificacao_LLM'] = df_valid['Classificacao_LLM'].astype(str).str.strip().str.lower()
        df_valid['Classificacao_LLM'] = df_valid['Classificacao_LLM'].map(lambda x: mapeamento.get(x, "Desconhecido"))
        
        df_final = df_valid[df_valid[TRUE_LABEL_COLUMN_NAME].isin(labels_order)].copy()
        
        y_true = df_final[TRUE_LABEL_COLUMN_NAME]
        y_pred = df_final['Classificacao_LLM']
        
        print(f"\nMétricas (Baseado em {len(df_final)} amostras válidas processadas)")
        
        try:
            print(classification_report(y_true, y_pred, labels=labels_order, zero_division=0))
        except Exception as e:
            print(f"Erro ao gerar classification_report: {e}")

        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels_order, yticklabels=labels_order,
                    annot_kws={"size": 12})
        
        plt.xlabel('Previsto (LLM)', fontsize=14)
        plt.ylabel('Verdadeiro (Gabarito)', fontsize=14)
        plt.title('Matriz de Confusão - Llama 4 Scout (Groq)', fontsize=16)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        confusion_matrix_filename = 'matriz_confusao_llama_scout.png'
        plt.savefig(confusion_matrix_filename)
        print(f"\nMatriz de confusão salva como '{confusion_matrix_filename}'")
        plt.show()