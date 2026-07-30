# Synthetic Face Detection

Projeto de Trabalho de Conclusão de Curso (TCC) para classificação de imagens faciais reais e sintéticas utilizando técnicas de aprendizado profundo.

O objetivo é desenvolver um modelo capaz de identificar se uma imagem facial foi gerada artificialmente por modelos generativos de inteligência artificial ou capturada de uma pessoa real.

---

## Objetivo

Com o avanço dos modelos generativos de imagens, como GANs e modelos de difusão, tornou-se cada vez mais difícil distinguir imagens reais de imagens sintéticas.

Este projeto busca avaliar técnicas de visão computacional para classificação binária:

- **Classe 0:** Imagem real
- **Classe 1:** Imagem sintética gerada por IA

---

# Dataset

O dataset utilizado contém **16.000 imagens faciais** balanceadas:

| Classe | Quantidade |
|---|---:|
| Reais | 8.000 |
| Sintéticas | 8.000 |

Todas as imagens possuem:

- Resolução: 1024x1024 pixels
- Formato: JPEG
- Espaço de cor: RGB

## Imagens sintéticas

| Modelo | Quantidade |
|---|---:|
| StyleGAN2 | 2.000 |
| StyleGAN3 | 2.000 |
| Stable Diffusion 1.4 | 2.000 |
| Stable Diffusion 2.1 | 2.000 |

## Imagens reais

| Dataset | Quantidade |
|---|---:|
| FFHQ | 6.000 |
| CelebA-HQ | 2.000 |