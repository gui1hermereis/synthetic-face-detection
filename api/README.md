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
    model_metadata.json
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
4. detecta se existe exatamente um rosto humano
5. rejeita imagem tremida ou com contraste ruim
6. aplica o preprocessamento do treinamento:
   `RGB -> Resize(256x256) -> ToTensor -> Normalize(ImageNet)`
7. executa a inferencia
8. retorna classe, confianca, probabilidades e metadados do modelo

## Endpoint de health

`GET /api/v1/health`

## Exemplo de requisicao

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/images/analyze" \
  -H "X-API-Key: SUA_API_KEY" \
  -F "image=@/caminho/para/imagem.jpg"
```

## Como executar

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

- O arquivo de pesos esperado por padrao esta em `./models/resnet50_rgb_256.pth`.
- O modelo precisa ter sido salvo com a mesma arquitetura do notebook.
- Preencha `models/model_metadata.json` com as metricas reais do treinamento para expor accuracy, precision, recall e f1 na resposta.
