import os
import pandas as pd

# Caminho exato que vi no seu print
base_path = "dataset_split/dataset_split"
categorias = ["test"]
extensoes_validas = ('.jpg', '.jpeg', '.png')

dados = []

print("Buscando imagens nas pastas...")

for cat in categorias:
    caminho_cat = os.path.join(base_path, cat)
    
    if not os.path.exists(caminho_cat):
        print(f"Aviso: Pasta {caminho_cat} não encontrada.")
        continue

    # Percorre subpastas (caso existam pastas 'Healthy', 'DR', etc dentro de 'test')
    for root, dirs, files in os.walk(caminho_cat):
        for file in files:
            if file.lower().endswith(extensoes_validas):
                # Caminho relativo para o script de classificação encontrar
                full_path = os.path.join(root, file)
                
                # Extrai o label da pasta pai (ex: Healthy, Mild_DR) 
                # Se não houver subpastas, o label será 'test', 'train' ou 'val'
                label = os.path.basename(root) if os.path.basename(root) != cat else "Unknown"
                
                dados.append({
                    "Image_path": full_path,
                    "Label_Verdadeiro": label
                })

# Cria o DataFrame e adiciona o ID
df = pd.DataFrame(dados)
df.insert(0, 'ID', range(1, len(df) + 1))

# Salva o CSV que o seu código principal precisa
df.to_csv("input_image_llm.csv", index=False)
print(f"\nSucesso! {len(df)} imagens encontradas e salvas em 'input_image_llm.csv'.")