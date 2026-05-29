import os
import glob
import copy
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.transforms import (
    Compose, LoadImageD, ResizeD, ScaleIntensityD,
    EnsureChannelFirstD, RandFlipD, RandRotateD, RandAdjustContrastD,
    RandZoomd, RandGaussianNoised
)
from monai.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. MODELO: UNet Encoder para Classificação
# =====================================================================
# Utilizamos o encoder pré-treinado do UNet (ResNet34 + ImageNet)
# e adicionamos uma cabeça de classificação global no bottleneck.
# Isso permite comparar o poder de representação do UNet com o DenseNet.
class UNetClassifier(nn.Module):
    def __init__(self, num_classes=5, encoder_name="resnet34", dropout_rate=0.3):
        super().__init__()
        # Cria apenas o encoder do UNet via segmentation_models_pytorch
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights="imagenet"
        )

        # Tamanho do feature map no bottleneck do ResNet34
        encoder_out_channels = self.encoder.out_channels[-1]  # 512

        # Cabeça de classificação (dropout configurável pelo grid search)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(encoder_out_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        bottleneck = features[-1]      # Pega o feature map mais profundo
        return self.classifier(bottleneck)


# =====================================================================
# 2. PESOS DAS CLASSES (tratamento de desbalanceamento)
# =====================================================================
def calcular_pesos_classes(dados_treino, num_classes=5):
    labels = [item["label"] for item in dados_treino]
    contagem = Counter(labels)
    total_amostras = len(labels)

    pesos = []
    for i in range(num_classes):
        count_i: int = contagem.get(i, 1)  # evita divisão por zero
        peso = total_amostras / (num_classes * count_i)
        pesos.append(peso)

    print(f"Contagem de classes no treino: {dict(contagem)}")
    print(f"Pesos calculados (Loss): {[f'{p:.3f}' for p in pesos]}")
    return torch.tensor(pesos, dtype=torch.float)


# =====================================================================
# 3. FUNÇÃO DE INGESTÃO DE DADOS
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
            print(f"Aviso: pasta '{caminho_pasta}' não encontrada.")
            continue
        imagens = []
        for extensao in ['*.jpg', '*.jpeg', '*.png']:
            imagens.extend(glob.glob(os.path.join(caminho_pasta, extensao)))
        for caminho_imagem in imagens:
            arquivos_dados.append({"image": caminho_imagem, "label": label})
    return arquivos_dados


# =====================================================================
# 3. CONFIGURAÇÃO DE DIRETÓRIOS
# =====================================================================
# Ajuste o caminho abaixo para apontar para o seu dataset_split
DIRETORIO_BASE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset_split', 'dataset_split')

train_files = criar_dicionario_dados(os.path.join(DIRETORIO_BASE, "train"))
val_files   = criar_dicionario_dados(os.path.join(DIRETORIO_BASE, "val"))
test_files  = criar_dicionario_dados(os.path.join(DIRETORIO_BASE, "test"))

print(f"Total de imagens -> Treino: {len(train_files)} | Validação: {len(val_files)} | Teste: {len(test_files)}")

# =====================================================================
# 5. PIPELINE DE TRANSFORMAÇÕES (AUGMENTATION MELHORADO)
# =====================================================================
train_transforms = Compose([
    LoadImageD(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(224, 224)),
    ScaleIntensityD(keys=["image"]),
    RandFlipD(keys=["image"], prob=0.5, spatial_axis=0),
    RandFlipD(keys=["image"], prob=0.5, spatial_axis=1),
    RandRotateD(keys=["image"], range_x=0.2, prob=0.5),
    RandZoomd(keys=["image"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
    RandAdjustContrastD(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1)
])

val_transforms = Compose([
    LoadImageD(keys=["image"]),
    EnsureChannelFirstD(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(224, 224)),
    ScaleIntensityD(keys=["image"]),
])

# =====================================================================
# 6. CONFIGURAÇÃO BASE
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinando utilizando: {device}")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "melhor_modelo_unet.pth")

# Pesos para tratamento do desbalanceamento de classes
pesos_classes = calcular_pesos_classes(train_files).to(device)

# =====================================================================
# 7. FUNÇÃO DE PLOT DE CURVAS
# =====================================================================
def plotar_curvas(historia, params=None):
    """Salva curva de loss e curva de acurácia em arquivos separados."""
    epochs_range = range(1, len(historia["train_loss"]) + 1)
    titulo_extra = f" (lr={params['lr']:.0e}, dropout={params['dropout_rate']})" if params else ""

    # --- Curva de Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, historia["train_loss"], label="Treino", marker='o', markersize=3)
    plt.plot(epochs_range, historia["val_loss"],   label="Validação", marker='s', markersize=3)
    plt.title(f"Curva de Loss — UNet{titulo_extra}")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(RESULTS_DIR, "curva_loss_unet.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Curva de loss salva em '{loss_path}'")

    # --- Curva de Acurácia ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, historia["train_acc"], label="Treino",    marker='o', markersize=3)
    plt.plot(epochs_range, historia["val_acc"],   label="Validação", marker='s', markersize=3)
    plt.title(f"Curva de Acurácia — UNet{titulo_extra}")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(RESULTS_DIR, "curva_acuracia_unet.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"Curva de acurácia salva em '{acc_path}'")


# =====================================================================
# 8. LOOP DE TREINAMENTO E VALIDAÇÃO
# =====================================================================
def treinar_modelo(lr=1e-4, dropout_rate=0.3, batch_size=16, epochs=50, save_path=None, early_stop_patience=10):
    """Treina o UNetClassifier com os hiperparâmetros fornecidos.
    Aplica Early Stopping para abortar o treino se a Loss de Validação estacionar.
    Retorna (melhor val_loss, melhores pesos, histórico de métricas)."""
    if save_path is None:
        save_path = MODEL_SAVE_PATH

    model = UNetClassifier(num_classes=5, encoder_name="resnet34",
                           dropout_rate=dropout_rate).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=pesos_classes)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    train_loader = DataLoader(
        Dataset(data=train_files, transform=train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        Dataset(data=val_files, transform=val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    melhor_loss = float('inf')
    melhor_pesos = None
    epochs_no_improve = 0
    historia = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"\n--- Época {epoch+1}/{epochs} | lr={lr} | dropout={dropout_rate} | batch={batch_size} ---")

        # --- TREINO ---
        model.train()
        train_loss, steps, train_corretos, train_total = 0, 0, 0, 0
        for batch in train_loader:
            steps += 1
            inputs = batch["image"].to(device)
            labels = batch["label"].clone().detach().to(dtype=torch.long, device=device)

            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_corretos += (preds == labels).sum().item()
            train_total += labels.size(0)

        media_train_loss = train_loss / steps
        train_acc = train_corretos / train_total
        print(f"Loss de Treino: {media_train_loss:.4f} | Acurácia Treino: {train_acc:.4f}")

        # --- VALIDAÇÃO ---
        model.eval()
        val_loss, val_steps, val_corretos, val_total = 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                val_steps += 1
                inputs = batch["image"].to(device)
                labels = batch["label"].clone().detach().to(dtype=torch.long, device=device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                val_corretos += (preds == labels).sum().item()
                val_total += labels.size(0)

        media_val_loss = val_loss / val_steps
        val_acc = val_corretos / val_total
        print(f"Loss de Validação: {media_val_loss:.4f} | Acurácia Validação: {val_acc:.4f}")

        historia["train_loss"].append(media_train_loss)
        historia["val_loss"].append(media_val_loss)
        historia["train_acc"].append(train_acc)
        historia["val_acc"].append(val_acc)

        sched.step(media_val_loss)

        if media_val_loss < melhor_loss:
            melhor_loss = media_val_loss
            melhor_pesos = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("=> Novo melhor modelo temporário encontrado nessa época!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\n🛑 EARLY STOPPING ATIVADO! Nenhuma melhoria da Validação por {early_stop_patience} épocas.")
                print(f"Evitando Overfitting severo. Abortando treinamento antecipadamente na época {epoch+1}...")
                break

    torch.save(melhor_pesos, save_path)
    print(f"\nModelo salvo em '{save_path}' (val_loss={melhor_loss:.4f})")
    return melhor_loss, melhor_pesos, historia


# =====================================================================
# 8. GRID SEARCH DE HIPERPARÂMETROS
# =====================================================================
def grid_search(epochs=50):
    """Testa combinações de learning_rate e dropout_rate.
    Salva o melhor modelo global ao final."""

    # Grade de hiperparâmetros
    param_grid = {
        "lr":           [1e-3, 1e-4, 1e-5],
        "dropout_rate": [0.2, 0.3, 0.5],
        "batch_size":   [16],
    }

    resultados = []
    melhor_global_loss = float('inf')
    melhor_global_pesos = None
    melhor_params = None
    melhor_historia = None

    combinacoes = [
        {"lr": lr, "dropout_rate": dr, "batch_size": bs}
        for lr in param_grid["lr"]
        for dr in param_grid["dropout_rate"]
        for bs in param_grid["batch_size"]
    ]

    print(f"\n{'='*60}")
    print(f"GRID SEARCH — {len(combinacoes)} combinações x {epochs} épocas")
    print(f"{'='*60}\n")

    for i, params in enumerate(combinacoes):
        print(f"\n[{i+1}/{len(combinacoes)}] Testando: {params}")
        temp_path = os.path.join(RESULTS_DIR, f"temp_gs_{i}.pth")

        val_loss, pesos, historia = treinar_modelo(
            lr=params["lr"],
            dropout_rate=params["dropout_rate"],
            batch_size=params["batch_size"],
            epochs=epochs,
            save_path=temp_path
        )

        resultados.append({**params, "val_loss": val_loss})

        if val_loss < melhor_global_loss:
            melhor_global_loss = val_loss
            melhor_global_pesos = torch.load(temp_path, map_location=device, weights_only=True)
            melhor_params = params
            melhor_historia = historia

        # Remove arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Salva o melhor modelo global
    torch.save(melhor_global_pesos, MODEL_SAVE_PATH)

    print(f"\n{'='*60}")
    print(f"RESULTADO DO GRID SEARCH")
    print(f"{'='*60}")
    print(f"Melhor val_loss: {melhor_global_loss:.4f}")
    print(f"Melhores parâmetros: {melhor_params}")
    print(f"\nTodos os resultados:")
    
    # Prepara salvamento do ranking em arquivo texto formatado
    tabela_path = os.path.join(RESULTS_DIR, "historico_grid_search_unet.txt")
    with open(tabela_path, "w", encoding="utf-8") as f:
        f.write("========== RANKING GRID SEARCH (UNet) ==========\n\n")
        
        for r in sorted(resultados, key=lambda x: x["val_loss"]):
            linha = f"lr={r['lr']:.0e} | dropout={r['dropout_rate']} | batch={r['batch_size']} => val_loss={r['val_loss']:.4f}"
            print(f"  {linha}")
            f.write(f"{linha}\n")
            
    print(f"\nTabela completa do Grid Search salva em '{tabela_path}'.\n")

    # Plota curvas do melhor modelo encontrado
    if melhor_historia:
        print("\nGerando curvas do melhor modelo...")
        plotar_curvas(melhor_historia, melhor_params)

    return melhor_params


# =====================================================================
# 9. AVALIAÇÃO DO MODELO
# =====================================================================
def avaliar_modelo(melhores_params=None):
    print("\nIniciando avaliação no conjunto de Teste...")

    # Recria o modelo com os melhores parâmetros do grid search (se disponível)
    dropout = melhores_params["dropout_rate"] if melhores_params else 0.3
    model = UNetClassifier(num_classes=5, encoder_name="resnet34",
                           dropout_rate=dropout).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()

    todas_previsoes = []
    todos_rotulos_reais = []
    # Mapeamento original (deve bater com o modelo treinado)
    nomes_classes = ["healthy", "mild", "moderate", "severe", "proliferate"]

    # Ordem de exibição: severe aparece por último
    display_order = [0, 1, 2, 4, 3]  # healthy, mild, moderate, proliferate, severe
    display_names = [nomes_classes[i] for i in display_order]

    test_loader = DataLoader(
        Dataset(data=test_files, transform=val_transforms),
        batch_size=16, shuffle=False, num_workers=0
    )

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            _, previsoes = torch.max(outputs, 1)
            todas_previsoes.extend(previsoes.cpu().numpy())
            todos_rotulos_reais.extend(labels.cpu().numpy())

    report_str = classification_report(
        todos_rotulos_reais, todas_previsoes,
        labels=display_order,
        target_names=display_names,
        zero_division=0
    )
    print("\n================ RELATÓRIO DE CLASSIFICAÇÃO ================")
    print(report_str)

    # Salva o relatório em TXT
    txt_path = os.path.join(RESULTS_DIR, 'relatorio_metricas_unet.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("================ RELATÓRIO DE CLASSIFICAÇÃO (UNet) ================\n\n")
        f.write(report_str)
    print(f"Relatório de métricas salvo em '{txt_path}'.")

    cm = confusion_matrix(todos_rotulos_reais, todas_previsoes)
    # Reordena linhas e colunas para exibir severe por último
    cm_display = cm[np.ix_(display_order, display_order)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names)
    plt.xlabel('Previsão do Modelo')
    plt.ylabel('Rótulo Real')
    plt.title('Matriz de Confusão - UNet Encoder (ResNet34)')
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, 'matriz_confusao_unet.png')
    plt.savefig(save_path, dpi=300)
    print(f"\nMatriz salva como '{save_path}'.")
    plt.show()


# =====================================================================
# 10. EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    # Apenas avaliação — usa o modelo salvo sem retreinar
    melhores_params = {"dropout_rate": 0.3}
    avaliar_modelo(melhores_params=melhores_params)
