# Synthetic Face Detection

Sistema web para analisar uma imagem facial e classificá-la como **real** ou **sintética**.

O padrão oficial de produção é:

- arquitetura: **ResNet34**;
- entrada: **RGB, 224×224**;
- classes: `fake` e `real`;
- detector facial: YuNet;
- decisão: `probability_fake >= decision_threshold`.

## Componentes

```text
synthetic-face-detection/
├── training/  # notebooks, dados e treinamento
├── models/    # artefato ResNet34 e modelo YuNet
├── api/       # Flask: validação, detecção e inferência
└── app/       # React: upload e exibição do resultado
```

## Fluxo de análise

1. O frontend permite selecionar somente arquivos JPG, JPEG ou PNG.
2. A API valida o arquivo real em memória: formato, integridade, tamanho e resolução.
3. O YuNet exige exatamente um rosto humano detectável.
4. O crop facial é rejeitado se não atingir os critérios configurados de desfoque ou contraste.
5. A imagem facial é convertida para RGB, redimensionada para `224×224`, normalizada com ImageNet e enviada à ResNet34.
6. A API retorna a decisão, probabilidades, confiança, limiar, qualidade e dados de detecção para a interface.

Uploads não são persistidos em disco. HTTPS é responsabilidade do ambiente de implantação/proxy reverso.

## Formatos aceitos

| Aceitos | Rejeitados |
| --- | --- |
| JPG (`.jpg`) | WEBP |
| JPEG (`.jpeg`) | PDF |
| PNG (`.png`) | TXT |

A API confere o formato do conteúdo, não apenas a extensão do nome do arquivo.

## Modelo e artefatos

A API usa, por padrão:

- `models/resnet34_224.pt`
- `models/face_detection_yunet_2023mar.onnx`

O artefato precisa declarar ResNet34, RGB, 224 px, as classes `fake`/`real`, normalização RGB e metadados de crop facial. Artefatos de outro tamanho, arquitetura ou quatro canais/FFT são recusados.

As métricas de avaliação do modelo — acurácia, precisão, recall, F1-score e matriz de confusão — estão no notebook oficial [training/notebooks/0.1_resnet34_224.ipynb](training/notebooks/0.1_resnet34_224.ipynb). Elas não são recalculadas pela API.

## Executar localmente

### API

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Configure `API_KEY` e, caso necessário, os caminhos `MODEL_WEIGHTS_PATH` e `FACE_DETECTOR_MODEL_PATH` em `api/.env`.

A API fica disponível em `http://127.0.0.1:5000/api/v1`.

### Frontend

```bash
cd app
npm install
cp .env.example .env
npm run dev
```

Em `app/.env`, defina `API_URL` como a base da API (por exemplo, `http://127.0.0.1:5000/api/v1`) e use a mesma `API_KEY` da API.

## Testes automatizados

Os testes não carregam o modelo pesado nem usam GPU: empregam fakes compatíveis com o contrato oficial.

```bash
cd api
.venv/bin/python -m pytest -q
```

A suíte cobre autenticação, formatos, corrupção, resolução, detecção de zero/múltiplos rostos, qualidade, classificação real/fake, limiar, contrato de resposta, confidencialidade em memória, erros internos seguros e contrato ResNet34/RGB/224.

## Benchmarks manuais

Os benchmarks ficam fora do `pytest`, pois medem uma API realmente em execução e não definem metas artificiais.

```bash
cd api
.venv/bin/python benchmarks/benchmark_single.py \
  --url http://127.0.0.1:5000 \
  --api-key SUA_API_KEY \
  --image /caminho/face.jpg \
  --requests 20
```

```bash
cd api
.venv/bin/python benchmarks/benchmark_concurrent.py \
  --url http://127.0.0.1:5000 \
  --api-key SUA_API_KEY \
  --image /caminho/face.jpg \
  --requests 50 \
  --concurrency 10
```

Os dois scripts reportam sucessos, falhas, mínimo, máximo, média, mediana, p95 e throughput. O segundo recebe concorrência configurável.

Para detalhes do serviço e do frontend, consulte [api/README.md](api/README.md) e [app/README.md](app/README.md).
