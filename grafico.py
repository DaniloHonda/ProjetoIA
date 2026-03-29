import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 1. Carrega o seu CSV
# (Substitua 'dados_teste.csv' pelo nome do arquivo onde você salvou esses dados)
df = pd.read_csv("classificacao_image_llama90b.csv")

# 2. Mapeamento para garantir que os nomes fiquem idênticos
mapeamento_gabarito = {
    "healthy": "Healthy",
    "mild": "Mild DR",
    "moderate": "Moderate DR",
    "proliferate": "Proliferative DR",
    "severe": "Severe DR"
}

# Aplica o mapeamento na coluna verdadeira
df['Label_Verdadeiro'] = df['Label_Verdadeiro'].astype(str).str.strip().str.lower()
df['Label_Verdadeiro'] = df['Label_Verdadeiro'].map(mapeamento_gabarito)

# Garante a limpeza da coluna do LLM
df['Classificacao_LLM'] = df['Classificacao_LLM'].astype(str).str.strip()

# 3. Define a ordem das classes
labels_order = ["Healthy", "Mild DR", "Moderate DR", "Proliferative DR", "Severe DR"]

# 4. Gera o Relatório de Classificação no Terminal
print("Métricas de Avaliação:\n")
print(classification_report(df['Label_Verdadeiro'], df['Classificacao_LLM'], labels=labels_order, zero_division=0))

# 5. Gera e Salva a Matriz de Confusão
cm = confusion_matrix(df['Label_Verdadeiro'], df['Classificacao_LLM'], labels=labels_order)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_order, yticklabels=labels_order,
            annot_kws={"size": 12})

plt.xlabel('Previsto (LLM)', fontsize=14)
plt.ylabel('Verdadeiro (Gabarito)', fontsize=14)
plt.title('Matriz de Confusão - Amostra de Teste', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Salva a imagem
nome_imagem = 'matriz_confusao_amostra.png'
plt.savefig(nome_imagem)
print(f"\nGráfico salvo com sucesso como '{nome_imagem}'!")
plt.show()