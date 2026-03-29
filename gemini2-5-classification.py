import pandas as pd

from google import genai

import time

import sys

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

import os



# CONFIGURAÇÃO

GOOGLE_API_KEY = "AIzaSyB9pzlIWPwmIRlZ24QiZKDeecCKywaGYzw"

if not GOOGLE_API_KEY:

    print("Erro: GOOGLE_API_KEY não está configurada.")

    sys.exit()



client = genai.Client(api_key=GOOGLE_API_KEY)



# Caminho do CSV

input_csv = "input_image_llm.csv"

output_csv = "classificacao_image_2-5pro.csv"



# VARIÁVEIS DE CONTROLE

MAX_CALLS_PER_RUN = 350

api_calls_made_this_run = 0



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

        print(f"Arquivo '{output_csv}' encontrado, mas está vazio ou corrompido. Começando do zero.")

    except Exception as e:

        print(f"Erro ao ler cache '{output_csv}': {e}. Começando do zero.")

else:

    print(f"Arquivo de saída '{output_csv}' não encontrado. Começando do zero.")



# Salva o total de chamadas já feitas (carregadas do cache)

total_ja_processados = len(existing_classifications)





# 2. LÊ O ARQUIVO DE ENTRADA PRINCIPAL

try:

    df_input = pd.read_csv(input_csv)

except FileNotFoundError:

    print(f"Erro: Arquivo de entrada '{input_csv}' não encontrado.")

    sys.exit()



# INDICADOR DE STATUS

total_a_processar = len(df_input) - total_ja_processados



print(f"\nStatus do Processamento")

print(f"Total de Imagens: {len(df_input)}")

print(f"Já Processados (Cache): {total_ja_processados}")

print(f"Restantes a Processar: {total_a_processar}")

print(f"---------------------------------")

print(f"Limite de NOVAS chamadas para esta execução: {MAX_CALLS_PER_RUN}\n")



# Cria lista para armazenar resultados

resultados = []



# 3. LOOP PRINCIPAL MODIFICADO

for i, row in df_input.iterrows():

    image_id = row["ID"]
    file_path = row["Image_path"]
    



    # 1. Já foi classificado

    if image_id in existing_classifications:

        classificacao = existing_classifications[image_id]

        resultados.append(classificacao)


   

    # 2. Limite de hoje atingido

    elif api_calls_made_this_run >= MAX_CALLS_PER_RUN:

        resultados.append("Pending")

        print(f"({i+1}/{len(df_input)}) ID={image_id}: Limite de API ({MAX_CALLS_PER_RUN}) atingido. Marcado como 'Pending'.")

   

    # 3. Chamar API

    else:

        print(f"Classificando imagem {i+1}/{len(df_input)}: ID={image_id} (Chamada {api_calls_made_this_run + 1}/{MAX_CALLS_PER_RUN})")

       

        try:
            print(f"Carregando imagem {file_path}...")
            my_file = client.files.upload(file=file_path)
            response = client.models.generate_content(

                model="gemma-3-27b-it",
                # Modelo sabe interpretar melhor em inglês, por isso optei por seguir com o prompt dessa maneira
                contents=[my_file, """You are an expert model in detecting diabetic retinopathy.
                            Classify the attached medical image into one of the following categories:
                                - Healthy
                                - Mild DR
                                - Moderate DR
                                - Proliferative DR
                                - Severe DR

                            Answer **ONLY** with the exact name of the category. Do not add any explanation."""]

            )



            classificacao = response.text.strip()

            print(f"Resultado: {classificacao}\n")

            resultados.append(classificacao)

            api_calls_made_this_run += 1

           

            time.sleep(3)



        except Exception as e:

            print(f"Erro ao processar ID={image_id}: {str(e)}\n")

            resultados.append(f"Erro: {str(e)}")

            api_calls_made_this_run += 1

            time.sleep(6)



# 4. SALVAR RESULTADOS

df_input["Classificacao_LLM"] = resultados



print(f"\n{api_calls_made_this_run} novas chamadas de API realizadas nesta execução.")

print(f"Total de chamadas acumuladas (salvas): {total_ja_processados + api_calls_made_this_run}")

print(f"Resultados salvos em {output_csv}")

df_input.to_csv(output_csv, index=False)





# 5. GERAR MÉTRICAS

print("\nGerando métricas de avaliação...")



TRUE_LABEL_COLUMN_NAME = 'Label_Verdadeiro'



if TRUE_LABEL_COLUMN_NAME not in df_input.columns:

    print(f"Erro: A coluna '{TRUE_LABEL_COLUMN_NAME}' não foi encontrada no CSV.")

    print(f"Por favor, ajuste a variável 'TRUE_LABEL_COLUMN_NAME' no código.")

else:

    labels_order = [

        "Healthy",

        "Mild DR",

        "Moderate DR",

        "Proliferative DR",

        "Severe DR"

    ]

   

    # 1. Filtra apenas classificações BEM-SUCEDIDAS

    df_valid = df_input[

        pd.notna(df_input['Classificacao_LLM']) &

        ~df_input['Classificacao_LLM'].str.startswith('Erro:') &

        (df_input['Classificacao_LLM'] != 'Pending')

    ].copy()

   

    if len(df_valid) == 0:

        print("Nenhuma amostra válida encontrada para gerar métricas. (Podem estar todas pendentes ou com erro).")

    else:

        df_valid[TRUE_LABEL_COLUMN_NAME] = df_valid[TRUE_LABEL_COLUMN_NAME].astype(str).str.strip()

        df_valid['Classificacao_LLM'] = df_valid['Classificacao_LLM'].astype(str).str.strip()

       

        y_true = df_valid[TRUE_LABEL_COLUMN_NAME]

        y_pred = df_valid['Classificacao_LLM']

       

        print(f"\nMétricas (Baseado em {len(df_valid)}/{len(df_input)} amostras válidas processadas)")

       

        # 2. Relatório de Classificação

        try:

            print(classification_report(y_true, y_pred, labels=labels_order, zero_division=0))

        except Exception as e:

            print(f"Erro ao gerar classification_report: {e}")



        # 3. Matriz de Confusão (Dados)

        cm = confusion_matrix(y_true, y_pred, labels=labels_order)

       

        # 4. Plotar a Matriz

        plt.figure(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',

                    xticklabels=labels_order, yticklabels=labels_order,

                    annot_kws={"size": 12})

       

        plt.xlabel('Previsto (LLM)', fontsize=14)

        plt.ylabel('Verdadeiro (Gabarito)', fontsize=14)

        plt.title('Matriz de Confusão - Classificação LLM', fontsize=16)

        plt.xticks(rotation=45)

        plt.yticks(rotation=0)

        plt.tight_layout()

       

        # 5. Salva a imagem

        confusion_matrix_filename = 'matriz_confusao.png'

        plt.savefig(confusion_matrix_filename)

        print(f"\nMatriz de confusão salva como '{confusion_matrix_filename}'")

       

        plt.show()