# Complemento — revisão humana direcionada da auditoria perceptual

**Base:** `training/analysis/auditoria_duplicatas/revisao_pares_entre_particoes.csv` e relatórios de auditoria já existentes.  
**Gerado em (UTC):** 2026-10-19T18:53:01.938994+00:00.

## Histórico preservado

- Imagens originais: **16.000**; crops válidos: **15.960**.
- Candidatos dHash totais no relatório anterior: **2.374**.
- Candidatos entre partições existentes no CSV: **1.224**, dos quais **988** envolvem teste.
- Pares com dHash de 0 a 2: **18**.

## Procedimento e cobertura humana

- Prioritários dHash 0–2 com decisão registrada: **18/18**.
- Complementares selecionados: **12**, dos quais **0** com decisão registrada.
- Total entre partições com decisão: **18/1.224**.
- Pares revisados envolvendo teste: **15**.
- Pares pendentes de revisão: **1.206**.
- Classificados como imagens distintas: **18**.
- Confirmados como mesma imagem: **0**.
- Marcados como mesma pessoa em imagens diferentes: **0**.
- Inconclusivos: **0**.
- Grupos conectados por pares candidatos: **164** (não equivalem a identidades).
- Grupos com ao menos uma relação envolvendo teste: **118**.

**Critério:** priorizar pares dHash 0–2; complementar com candidatos de dHash 3 envolvendo o teste, menor pHash e diversidade de grupos conectados. As classificações são humanas, registradas por par, revisor e data no CSV.

## Interpretação e limites

A revisão documenta somente as relações efetivamente inspecionadas; pares ainda pendentes não são considerados duplicatas nem descartados como semelhantes. As distâncias dHash/pHash não provam identidade. Um grupo conectado por pares candidatos não significa que suas imagens sejam da mesma pessoa. Não se demonstra ausência completa de sobreposição visual ou de identidade entre partições. O conjunto de teste e os checkpoints não foram alterados por estas etapas.

**Encaminhamento:** se houver par confirmado envolvendo teste ou situação inconclusiva relevante, avaliar o impacto e discuti-lo com o orientador; não alterar silenciosamente o protocolo nem os resultados originais.

**Evidências:** `revisao_pares_entre_particoes.csv`, `triagem_complementar_12.csv` (quando existir), `agrupamentos_candidatos_revisao.csv`, pranchas de comparação e o histórico `relatorio_auditoria_duplicatas.json`.
