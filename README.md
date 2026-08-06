# Synthetic Face Detection

Projeto dividido em tres modulos principais para treinamento, produto web e inferencia via API.

## Visao geral

O objetivo do projeto e classificar imagens faciais como reais ou sinteticas. A estrutura foi separada para que cada parte tenha responsabilidade clara:

- `training/`: desenvolvimento, experimentacao e treinamento do modelo
- `api/`: servico Flask responsavel por validar a imagem e executar a inferencia
- `app/`: interface web em React para envio da imagem e visualizacao do resultado

## Estrutura principal

```text
synthetic-face-detection/
  api/
  app/
  training/
  README.md
```

## Modulo `training`

Responsavel pelo ciclo de pesquisa e evolucao do modelo.

Contem:

- `notebooks/`: exploracao e experimentos
- `scripts/`: utilitarios de treino e dados
- `src/`: codigo Python reutilizavel do pipeline de treinamento
- `models/`: pesos `.pth` gerados no treinamento
- `outputs/`: metricas, graficos, logs e artefatos de avaliacao
- `requirements.txt`: dependencias do ambiente de treino
- `.gitignore`: regras locais para pesos, outputs e caches

Ponto importante:

- o modelo final usado em producao deve ser copiado para `api/models/`

## Modulo `api`

Responsavel por expor a inferencia do modelo em uma API Flask.

Contem:

- autenticacao por `X-API-Key`
- validacao da imagem recebida
- deteccao de rosto unico
- checagem de qualidade da imagem
- preprocessamento compativel com o treinamento
- inferencia com o modelo salvo em `api/models/`
- testes automatizados da API

Arquivos principais:

- `api/README.md`
- `api/requirements.txt`
- `api/.env`
- `api/models/`
- `api/src/`
- `api/tests/`

## Modulo `app`

Responsavel pela interface web do projeto.

Contem:

- upload da imagem
- leitura de `VITE_API_KEY` e `VITE_API_BASE_URL` por `.env`
- envio autenticado para a API
- exibicao da classificacao, confianca e probabilidades

Arquivos principais:

- `app/README.md`
- `app/package.json`
- `app/.env.example`
- `app/src/`

## Fluxo geral

1. o modelo e treinado em `training/`
2. o peso final `.pth` e colocado em `api/models/`
3. a API carrega esse modelo e expoe a inferencia
4. o frontend envia a imagem para a API e mostra o resultado ao usuario

## Onde configurar cada parte

- treino: `training/requirements.txt`
- API: `api/.env` e `api/requirements.txt`
- frontend: `app/.env` e `app/package.json`
