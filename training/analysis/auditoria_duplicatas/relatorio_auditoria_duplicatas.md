# Auditoria independente de duplicatas — 2026-10-18

**Escopo:** crops do experimento ResNet34, partições source_holdout/seed 42; os seis experimentos utilizam o mesmo manifesto/partições documentados.

- Arquivos originais: **16,000**; crops válidos: **15,960**.
- Repartição preservada: treino **9,576**, validação **2,394**, teste **3,990**.
- Integridade dos crops frente ao SHA-256 do treinamento: **0 divergências**.
- Arquivos originais com SHA-256 repetido: **0** (ocorrências, não pares).
- Crops com SHA-256 repetido: **0** (ocorrências, não pares).
- Triagem perceptual dHash 64 bits com distância ≤ 5: **2,374 pares candidatos**, sendo **1,224 entre partições diferentes**.
- Candidatos entre partições com dHash de 0 a 2: **18**.
- Revisão humana: **0 de 1,224 pares analisados**; pendentes **1,224**, inconclusivos **0**.
- Confirmados pela inspeção: mesma imagem **0**; mesma pessoa em imagens diferentes **0**.
- Metadados confiáveis de identidade presentes em 0 registros; a ausência desses metadados **não** demonstra independência de identidades.

**Situação:** REVISÃO HUMANA PENDENTE.

**Interpretação:** similaridade perceptual isolada não demonstra duplicação nem vazamento entre partições. A cobertura da auditoria de identidade depende de metadados confiáveis. Quando existem pares confirmados, avaliar impacto nos resultados e documentar decisão com o orientador. O conjunto de teste não foi alterado por este procedimento.

**Evidências:** `duplicatas_exatas_originais.csv`, `duplicatas_exatas_crops.csv`, `candidatos_dhash_todos.csv`, `revisao_pares_entre_particoes.csv`, pranchas locais e manifestos originais do GitHub.
