# Treinamento

Módulo responsável pela preparação dos dados, treinamento, experimentação e avaliação do modelo de classificação de faces reais e sintéticas.

## Padrão oficial

O modelo final selecionado segue este padrão:

- arquitetura: **ResNet34**;
- entrada: **RGB, 224×224**;
- classes: `fake` e `real`;
- sem FFT;
- detecção facial: **YuNet**;
- exatamente uma face por imagem;
- crop quadrado com margem de **15%**;
- estratégia de avaliação: `source_holdout`.

O notebook oficial é:

```text
notebooks/0.1_resnet34_224.ipynb
```

Os demais notebooks são experimentais e foram utilizados para comparação de arquiteturas, resoluções e técnicas.

## Estrutura

```text
training/
├── notebooks/
│   ├── 0.0_analise_exploratoria.ipynb
│   ├── 0.1_resnet34_224.ipynb
│   ├── 0.1_resnet34_224_fft.ipynb
│   ├── 0.2_resnet18_224.ipynb
│   ├── 0.2_resnet50_224.ipynb
│   ├── 0.3_resnet50_512.ipynb
│   └── 0.3_resnet_256.ipynb
├── scripts/
│   ├── download_dataset.py
│   ├── face_detector.py
│   └── loader_dataset.py
├── src/
├── analysis/
├── dataset/                         # imagens originais, não versionadas
├── dataset_faces_224_m15_yunet_v1/ # crops processados, não versionados
├── requirements.txt
└── README.md
```

## Preparar o ambiente

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Para utilizar GPU, instale `torch` e `torchvision` de acordo com a versão de CUDA disponível no ambiente.

## Dados e YuNet

As imagens originais são armazenadas em:

```text
training/dataset/
```

Os dados são organizados entre as classes:

```text
fake
real
```

O pré-processamento utiliza o detector facial YuNet:

```text
models/face_detection_yunet_2023mar.onnx
```

Os crops processados são armazenados em:

```text
training/dataset_faces_224_m15_yunet_v1/
```

O processo utiliza:

```text
YuNet
→ exatamente uma face
→ crop quadrado
→ margem 0.15
→ resize 224×224
```

## Pipeline oficial

Execute o notebook:

```bash
jupyter lab notebooks/0.1_resnet34_224.ipynb
```

O pipeline:

1. detecta uma única face com YuNet;
2. gera o crop facial quadrado com margem de 15%;
3. verifica dados inválidos e duplicados;
4. separa treino, validação e teste;
5. treina a ResNet34 com imagens RGB normalizadas pelo ImageNet;
6. seleciona o checkpoint utilizando a validação;
7. seleciona o limiar de decisão utilizando a validação;
8. avalia o conjunto de teste;
9. exporta o modelo final.

O conjunto de teste não é utilizado para escolher arquitetura, época ou limiar.

## Limiar de decisão

O limiar é selecionado no conjunto de validação utilizando F-beta com:

```text
beta = 2
classe positiva = fake
```

O limiar obtido para o modelo final foi:

```text
P(fake) >= 0.20
```

A regra de decisão é:

```text
P(fake) >= 0.20 → fake
P(fake) < 0.20  → real
```

## Avaliação

As métricas utilizadas incluem:

- acurácia;
- precisão;
- recall;
- F1-score;
- ROC-AUC;
- PR-AUC;
- matriz de confusão.

Os resultados completos estão registrados no notebook:

```text
notebooks/0.1_resnet34_224.ipynb
```

## Artefato final

O treinamento exporta:

```text
models/resnet34_224.pt
models/resnet34_224.json
```

O artefato contém:

- `state_dict`;
- arquitetura;
- tamanho de entrada;
- classes;
- normalização RGB;
- limiar de decisão;
- configuração do crop facial.

O modelo final utiliza **ResNet34 RGB 224×224 sem FFT**.

## Reprodutibilidade

Ao comparar experimentos:

- preserve a `SEED`;
- mantenha registradas as fontes utilizadas em cada partição;
- preserve os hiperparâmetros do treinamento;
- utilize o mesmo pré-processamento;
- escolha checkpoint e limiar somente com o conjunto de validação;
- mantenha o conjunto de teste separado até a avaliação final.