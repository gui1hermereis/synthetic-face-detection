# API Flask

API organizada para inferencia de deteccao de rosto sintetico com:

- autenticacao via `X-API-Key`
- validacao de arquivo e resolucao
- deteccao de exatamente um rosto humano
- checagem de qualidade da imagem
- preprocessamento identico ao notebook `training/notebooks/train_rgb.ipynb`
- inferencia com `ResNet50`
- testes automatizados com `pytest`

## Estrutura

```text
api/
  app.py
  .env
  .env.example
  models/
    resnet50_rgb_256.pth
  src/
    config/
    errors/
    routes/
    security/
    services/
  tests/
```

## Endpoint principal

`POST /api/v1/images/analyze`

Fluxo:

1. valida `X-API-Key`
2. recebe a imagem no campo multipart `image`
3. valida tamanho e resolucao
4. detecta se existe exatamente um rosto humano com YuNet
5. recorta o rosto detectado, incluindo uma margem configuravel
6. rejeita o recorte facial se estiver tremido ou com contraste ruim
7. aplica o preprocessamento do treinamento:
   `RGB -> Resize(256x256) -> ToTensor -> Normalize(ImageNet)`
8. executa a inferencia
9. retorna classe, confianca, probabilidades e dados do modelo carregado

## Endpoint de health

`GET /api/v1/health`

## Exemplo de requisicao

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/images/analyze" \
  -H "X-API-Key: SUA_API_KEY" \
  -F "image=@/caminho/para/imagem.jpg"
```

## Como executar

Antes de iniciar a API, baixe o modelo YuNet oficial e salve-o em
`api/models/face_detection_yunet_2023mar.onnx`:

```bash
mkdir -p models
curl -fL -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

Execute esse comando a partir da pasta `api/`. O modelo nao e versionado pelo
Git; em producao, disponibilize-o no mesmo caminho ou configure
`FACE_DETECTOR_MODEL_PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Como rodar os testes

```bash
pip install pytest
pytest tests -q
```

## Observacoes importantes

- A API carrega o primeiro arquivo `.pth` encontrado em `./models/`.
- O modelo precisa ter sido salvo com a mesma arquitetura do notebook.
