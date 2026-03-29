import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.transforms import (
    Compose, LoadImageD, ResizeD, ScaleIntensityD, 
    EnsureChannelFirstD, RandFlipD, RandRotateD, RandAdjustContrastD
)
from monai.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. FUNÇÃO DE INGESTÃO DE DADOS
# =====================================================================
def criar_dicionario_dados(diretorio_base):
    mapa_classes = {
        "healthy": 0,       
        "mild": 1,          
        "moderate": 2,      
        "severe": 3,        
        "proliferate": 4    
    }
    arquivos_dados = []
    for nome_pasta, label in mapa_classes.items():
        caminho_pasta = os.path.join(diretorio_base, nome_pasta)
        if not os.path.exists(caminho_pasta):
            continue
        imagens = []
        for extensao in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            imagens.extend(glob.glob(os.path.join(caminho_pasta, extensao)))
        for caminho_imagem in imagens:
            arquivos_dados.append({"image": caminho_imagem, "label": label})
    return arquivos_dados

# =====================================================================
# 2. CONFIGURAÇÃO DE DIRETÓRIOS E CARREGAMENTO
# =====================================================================
DIRETORIO_BASE = r"C:\Users\Luis Basacchi\Documents\Estudos\Faculdade\TCC\dataset_split" 

train_files = criar_dicionario_dados(os.path.join(DIRETORIO_BASE, "train"))
val_files = criar_dicionario_dados(os.path.join(DIRETORIO_BASE, "val"))
test_files = criar_dicionario_dados(os.path.join(DIRETORIO_BASE, "test"))

print(f"Total de imagens -> Treino: {len(train_files)} | Validação: {len(val_files)} | Teste: {len(test_files)}")

# =====================================================================
# 3. PIPELINE DE TRANSFORMAÇÕES (COM AJUSTE DE CONTRASTE)
# =====================================================================
train_transforms = Compose([
    LoadImageD(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]), 
    ResizeD(keys=["image"], spatial_size=(224, 224)), 
    ScaleIntensityD(keys=["image"]), 
    RandFlipD(keys=["image"], prob=0.5, spatial_axis=0), 
    RandRotateD(keys=["image"], range_x=0.1, prob=0.5), 
    RandAdjustContrastD(keys=["image"], prob=0.5, gamma=(0.5, 2.0)) # Ajuda com fotos muito escuras/claras
])

val_transforms = Compose([
    LoadImageD(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(224, 224)),
    ScaleIntensityD(keys=["image"]),
])

# =====================================================================
# 4. CRIAÇÃO DOS DATALOADERS
# =====================================================================
train_loader = DataLoader(
    Dataset(data=train_files, transform=train_transforms), 
    batch_size=16, 
    shuffle=True, 
    num_workers=0
)

val_loader = DataLoader(
    Dataset(data=val_files, transform=val_transforms), 
    batch_size=16, 
    shuffle=False, 
    num_workers=0
)

# =====================================================================
# 5. TRANSFER LEARNING: MODELO, LOSS, OTIMIZADOR E SCHEDULER
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinando utilizando: {device}")

# Importa a DenseNet121 já pré-treinada com milhões de imagens (ImageNet)
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

# Adapta a última camada (classificador) para as nossas 5 classes de retinopatia
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 5)

model = model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# O Scheduler vai reduzir a taxa de aprendizado se o modelo parar de evoluir
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# =====================================================================
# 6. LOOP DE TREINAMENTO E VALIDAÇÃO
# =====================================================================
def treinar_modelo(epochs=25):
    melhor_loss_validacao = float('inf')
    
    for epoch in range(epochs):
        print(f"\n--- Época {epoch+1}/{epochs} ---")
        
        # --- TREINO ---
        model.train() 
        train_loss, passos_treino = 0, 0
        
        for batch in train_loader:
            passos_treino += 1
            inputs = batch["image"].to(device)
            labels = batch["label"].clone().detach().to(dtype=torch.long, device=device)
            
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = loss_function(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            train_loss += loss.item()
            
        media_train_loss = train_loss / passos_treino
        print(f"Loss de Treino: {media_train_loss:.4f}")
        
        # --- VALIDAÇÃO ---
        model.eval() 
        val_loss, passos_val = 0, 0
        
        with torch.no_grad(): 
            for batch in val_loader:
                passos_val += 1
                inputs = batch["image"].to(device)
                labels = batch["label"].clone().detach().to(dtype=torch.long, device=device)
                
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                
        media_val_loss = val_loss / passos_val
        print(f"Loss de Validação: {media_val_loss:.4f}")
        
        # Atualiza o Scheduler com base na loss de validação
        scheduler.step(media_val_loss)
        
        # Salva o melhor modelo
        if media_val_loss < melhor_loss_validacao:
            melhor_loss_validacao = media_val_loss
            torch.save(model.state_dict(), "melhor_modelo_retinopatia_transfer.pth")
            print("=> Novo melhor modelo salvo!")

# =====================================================================
# 7. AVALIAÇÃO DO MODELO
# =====================================================================
def avaliar_modelo():
    print("\nIniciando avaliação no conjunto de Teste...")
    model.load_state_dict(torch.load("melhor_modelo_retinopatia_transfer.pth", map_location=device, weights_only=True))
    model.eval()
    
    todas_previsoes = []
    todos_rotulos_reais = []
    nomes_classes = ["healthy", "mild", "moderate", "severe", "ploriferate"]
    
    test_loader = DataLoader(Dataset(data=test_files, transform=val_transforms), batch_size=16, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            _, previsoes = torch.max(outputs, 1)
            todas_previsoes.extend(previsoes.cpu().numpy())
            todos_rotulos_reais.extend(labels.cpu().numpy())
            
    print("\n================ RELATÓRIO DE CLASSIFICAÇÃO ================")
    print(classification_report(todos_rotulos_reais, todas_previsoes, target_names=nomes_classes))
    
    cm = confusion_matrix(todos_rotulos_reais, todas_previsoes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.xlabel('Previsão do Modelo')
    plt.ylabel('Rótulo Real')
    plt.title('Matriz de Confusão - Transfer Learning')
    plt.tight_layout()
    plt.savefig('matriz_de_confusao_transfer.png', dpi=300)
    print("\nMatriz salva como 'matriz_de_confusao_transfer.png'.")
    plt.show()

# =====================================================================
# 8. EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    treinar_modelo(epochs=25)
    avaliar_modelo()