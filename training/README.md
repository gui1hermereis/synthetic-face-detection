# Training

Modulo responsavel pelo ciclo de treinamento, avaliacao e experimentacao do modelo de deteccao de faces sinteticas.

## Estrutura

```text
training/
  notebooks/
  scripts/
  src/
  models/
  outputs/
  requirements.txt
  .gitignore
  README.md
```

## Responsabilidade de cada pasta

- `notebooks/`: exploracao, testes rapidos e validacoes manuais.
- `scripts/`: scripts executaveis, como download de dataset, treino e avaliacao.
- `src/`: codigo Python reutilizavel do pipeline de treinamento.
- `models/`: pesos exportados do treinamento, como `.pth`.
- `outputs/`: metricas, graficos, logs e artefatos gerados durante os experimentos.

## Como preparar o ambiente

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Convencao recomendada

- mover a logica repetida dos notebooks para `src/`
- deixar notebooks como camada de experimento, nao como fonte principal da regra de negocio
- salvar pesos treinados em `models/`
- salvar resultados de avaliacao em `outputs/`
- usar `scripts/` para fluxos reproduziveis de treino e avaliacao

## Observacao sobre a API

A API usa o modelo de producao dentro de `api/models/`. Se quiser publicar um novo peso treinado, copie o `.pth` final de `training/models/` para `api/models/`.
