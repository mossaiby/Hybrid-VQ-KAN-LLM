
### synth — all  (reference: vanilla_text:text)

| run:mode | n | acc % | 95% CI | text | latent | steps | vs ref: b/c | McNemar p |
|---|---|---|---|---|---|---|---|---|
| kanvq_hybrid_v2:hybrid+noise0.5 | 1100 | 12.5 | 10.7–14.6 | 79.6 | 59.4 | 139.0 | 196/31 | 1.7e-30 |
| kanvq_hybrid_v2:hybrid+noise2.0 | 1100 | 12.1 | 10.3–14.2 | 77.6 | 197.1 | 274.7 | 197/27 | 4.41e-33 |
| kanvq_hybrid_v2:hybrid+onestep | 1100 | 26.9 | 24.4–29.6 | 119.9 | 3.7 | 123.6 | 69/62 | 0.6 |
| kanvq_hybrid_v2:hybrid+quantize | 1100 | 26.9 | 24.4–29.6 | 57.2 | 63.6 | 120.8 | 74/67 | 0.614 |
| kanvq_hybrid_v2:hybrid | 1100 | 22.8 | 20.4–25.4 | 59.2 | 62.3 | 121.5 | 92/40 | 6.97e-06 |
| kanvq_hybrid_v2:text | 1100 | 27.1 | 24.5–29.8 | 120.6 | 0.0 | 120.6 | 72/67 | 0.735 |
| vanilla_hybrid_v2:hybrid+noise0.5 | 1100 | 23.5 | 21.0–26.0 | 63.2 | 63.6 | 126.7 | 109/64 | 0.000774 |
| vanilla_hybrid_v2:hybrid+noise2.0 | 1100 | 14.6 | 12.7–16.8 | 59.3 | 89.6 | 148.9 | 184/42 | 2.18e-22 |
| vanilla_hybrid_v2:hybrid+onestep | 1100 | 27.3 | 24.7–30.0 | 145.7 | 3.2 | 148.9 | 99/96 | 0.886 |
| vanilla_hybrid_v2:hybrid+quantize | 1100 | 31.9 | 29.2–34.7 | 64.6 | 67.0 | 131.6 | 49/97 | 8.77e-05 |
| vanilla_hybrid_v2:hybrid | 1100 | 29.0 | 26.4–31.8 | 64.3 | 63.9 | 128.2 | 59/75 | 0.195 |
| vanilla_hybrid_v2:text | 1100 | 33.2 | 30.5–36.0 | 132.0 | 0.0 | 132.0 | 44/106 | 4.4e-07 |
| kanvq_text_v2:text | 1100 | 24.0 | 21.6–26.6 | 120.1 | 0.0 | 120.1 | 81/42 | 0.000556 |
| vanilla_text:text | 1100 | 27.5 | 25.0–30.3 | 123.0 | 0.0 | 123.0 | — | — |
| sep_kan_plainhead_text:text | 1100 | 25.0 | 22.5–27.6 | 120.5 | 0.0 | 120.5 | 82/54 | 0.0203 |
| sep_swiglu_sphere_text:text | 1100 | 54.5 | 51.5–57.4 | 123.7 | 0.0 | 123.7 | 15/311 | 4.23e-73 |

Accuracy by depth (%):

| run:mode | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kanvq_hybrid_v2:hybrid+noise0.5 | 50 | 40 | 21 | 12 | 7 | 7 | 1 | 0 | 0 | 0 | 0 |
| kanvq_hybrid_v2:hybrid+noise2.0 | 50 | 40 | 20 | 9 | 7 | 6 | 1 | 0 | 0 | 0 | 0 |
| kanvq_hybrid_v2:hybrid+onestep | 90 | 75 | 49 | 37 | 20 | 15 | 8 | 1 | 0 | 0 | 1 |
| kanvq_hybrid_v2:hybrid+quantize | 91 | 75 | 48 | 38 | 18 | 16 | 7 | 1 | 1 | 0 | 1 |
| kanvq_hybrid_v2:hybrid | 90 | 67 | 40 | 28 | 12 | 11 | 3 | 0 | 0 | 0 | 0 |
| kanvq_hybrid_v2:text | 91 | 74 | 47 | 38 | 21 | 14 | 9 | 3 | 1 | 0 | 0 |
| vanilla_hybrid_v2:hybrid+noise0.5 | 89 | 65 | 41 | 24 | 13 | 11 | 8 | 4 | 3 | 0 | 0 |
| vanilla_hybrid_v2:hybrid+noise2.0 | 56 | 41 | 22 | 11 | 10 | 8 | 6 | 4 | 3 | 0 | 0 |
| vanilla_hybrid_v2:hybrid+onestep | 79 | 70 | 48 | 32 | 21 | 18 | 14 | 11 | 7 | 0 | 0 |
| vanilla_hybrid_v2:hybrid+quantize | 99 | 83 | 60 | 36 | 28 | 24 | 11 | 6 | 3 | 1 | 0 |
| vanilla_hybrid_v2:hybrid | 99 | 76 | 56 | 31 | 22 | 16 | 10 | 6 | 3 | 0 | 0 |
| vanilla_hybrid_v2:text | 99 | 82 | 65 | 36 | 24 | 23 | 15 | 11 | 9 | 1 | 0 |
| kanvq_text_v2:text | 83 | 66 | 47 | 28 | 22 | 9 | 6 | 2 | 1 | 0 | 0 |
| vanilla_text:text | 84 | 69 | 54 | 35 | 24 | 15 | 9 | 8 | 3 | 1 | 1 |
| sep_kan_plainhead_text:text | 82 | 69 | 44 | 33 | 24 | 11 | 3 | 5 | 3 | 0 | 1 |
| sep_swiglu_sphere_text:text | 95 | 92 | 85 | 88 | 76 | 67 | 56 | 32 | 8 | 0 | 0 |

### synth — ID d2-8  (reference: vanilla_text:text)

| run:mode | n | acc % | 95% CI | text | latent | steps | vs ref: b/c | McNemar p |
|---|---|---|---|---|---|---|---|---|
| kanvq_hybrid_v2:hybrid+noise0.5 | 700 | 19.7 | 16.9–22.8 | 71.5 | 39.2 | 110.7 | 183/31 | 1.98e-27 |
| kanvq_hybrid_v2:hybrid+noise2.0 | 700 | 19.0 | 16.3–22.1 | 75.3 | 169.0 | 244.3 | 184/27 | 6.54e-30 |
| kanvq_hybrid_v2:hybrid+onestep | 700 | 42.0 | 38.4–45.7 | 98.8 | 2.3 | 101.1 | 57/61 | 0.783 |
| kanvq_hybrid_v2:hybrid+quantize | 700 | 41.9 | 38.3–45.5 | 50.4 | 47.3 | 97.7 | 61/64 | 0.858 |
| kanvq_hybrid_v2:hybrid | 700 | 35.9 | 32.4–39.5 | 51.5 | 46.8 | 98.3 | 79/40 | 0.000445 |
| kanvq_hybrid_v2:text | 700 | 42.0 | 38.4–45.7 | 96.8 | 0.0 | 96.8 | 60/64 | 0.788 |
| vanilla_hybrid_v2:hybrid+noise0.5 | 700 | 35.9 | 32.4–39.5 | 50.5 | 50.5 | 101.0 | 99/60 | 0.00247 |
| vanilla_hybrid_v2:hybrid+noise2.0 | 700 | 22.0 | 19.1–25.2 | 47.9 | 83.0 | 130.9 | 174/38 | 5.45e-22 |
| vanilla_hybrid_v2:hybrid+onestep | 700 | 40.3 | 36.7–44.0 | 115.5 | 2.4 | 117.8 | 90/82 | 0.594 |
| vanilla_hybrid_v2:hybrid+quantize | 700 | 48.7 | 45.0–52.4 | 49.7 | 48.2 | 97.9 | 39/90 | 8.29e-06 |
| vanilla_hybrid_v2:hybrid | 700 | 44.3 | 40.6–48.0 | 50.3 | 48.7 | 99.0 | 49/69 | 0.0798 |
| vanilla_hybrid_v2:text | 700 | 49.1 | 45.5–52.8 | 96.9 | 0.0 | 96.9 | 36/90 | 1.64e-06 |
| kanvq_text_v2:text | 700 | 37.3 | 33.8–40.9 | 96.8 | 0.0 | 96.8 | 69/40 | 0.00704 |
| vanilla_text:text | 700 | 41.4 | 37.8–45.1 | 97.0 | 0.0 | 97.0 | — | — |
| sep_kan_plainhead_text:text | 700 | 38.0 | 34.5–41.7 | 96.8 | 0.0 | 96.8 | 72/48 | 0.0353 |
| sep_swiglu_sphere_text:text | 700 | 79.9 | 76.7–82.7 | 96.7 | 0.0 | 96.7 | 9/278 | 2.67e-70 |

### synth — OOD d>8  (reference: vanilla_text:text)

| run:mode | n | acc % | 95% CI | text | latent | steps | vs ref: b/c | McNemar p |
|---|---|---|---|---|---|---|---|---|
| kanvq_hybrid_v2:hybrid+noise0.5 | 400 | 0.0 | 0.0–1.0 | 93.8 | 94.8 | 188.6 | 13/0 | 0.000244 |
| kanvq_hybrid_v2:hybrid+noise2.0 | 400 | 0.0 | 0.0–1.0 | 81.8 | 246.2 | 328.0 | 13/0 | 0.000244 |
| kanvq_hybrid_v2:hybrid+onestep | 400 | 0.5 | 0.1–1.8 | 156.8 | 6.2 | 163.0 | 12/1 | 0.00342 |
| kanvq_hybrid_v2:hybrid+quantize | 400 | 0.8 | 0.3–2.2 | 69.1 | 92.1 | 161.2 | 13/3 | 0.0213 |
| kanvq_hybrid_v2:hybrid | 400 | 0.0 | 0.0–1.0 | 72.6 | 89.4 | 162.0 | 13/0 | 0.000244 |
| kanvq_hybrid_v2:text | 400 | 1.0 | 0.4–2.5 | 162.2 | 0.0 | 162.2 | 12/3 | 0.0352 |
| vanilla_hybrid_v2:hybrid+noise0.5 | 400 | 1.8 | 0.9–3.6 | 85.2 | 86.5 | 171.7 | 10/4 | 0.18 |
| vanilla_hybrid_v2:hybrid+noise2.0 | 400 | 1.8 | 0.9–3.6 | 79.3 | 101.2 | 180.5 | 10/4 | 0.18 |
| vanilla_hybrid_v2:hybrid+onestep | 400 | 4.5 | 2.9–7.0 | 198.6 | 4.7 | 203.3 | 9/14 | 0.405 |
| vanilla_hybrid_v2:hybrid+quantize | 400 | 2.5 | 1.4–4.5 | 90.8 | 99.9 | 190.7 | 10/7 | 0.629 |
| vanilla_hybrid_v2:hybrid | 400 | 2.2 | 1.2–4.2 | 88.8 | 90.4 | 179.2 | 10/6 | 0.454 |
| vanilla_hybrid_v2:text | 400 | 5.2 | 3.5–7.9 | 193.4 | 0.0 | 193.4 | 8/16 | 0.152 |
| kanvq_text_v2:text | 400 | 0.8 | 0.3–2.2 | 160.7 | 0.0 | 160.7 | 12/2 | 0.0129 |
| vanilla_text:text | 400 | 3.2 | 1.9–5.5 | 168.6 | 0.0 | 168.6 | — | — |
| sep_kan_plainhead_text:text | 400 | 2.2 | 1.2–4.2 | 161.9 | 0.0 | 161.9 | 10/6 | 0.454 |
| sep_swiglu_sphere_text:text | 400 | 10.0 | 7.4–13.3 | 171.0 | 0.0 | 171.0 | 6/33 | 1.43e-05 |

### gsm8k — all  (reference: vanilla_text:text)

| run:mode | n | acc % | 95% CI | text | latent | steps | vs ref: b/c | McNemar p |
|---|---|---|---|---|---|---|---|---|
| kanvq_hybrid_v2:hybrid | 1319 | 2.1 | 1.5–3.1 | 90.3 | 161.8 | 252.0 | 121/19 | 2.32e-19 |
| kanvq_hybrid_v2:text | 1319 | 2.7 | 1.9–3.7 | 216.0 | 0.0 | 216.0 | 117/22 | 7.58e-17 |
| vanilla_hybrid_v2:hybrid | 1319 | 2.2 | 1.5–3.1 | 46.8 | 145.3 | 192.1 | 119/18 | 1.89e-19 |
| vanilla_hybrid_v2:text | 1319 | 8.3 | 7.0–10.0 | 179.7 | 0.0 | 179.7 | 86/66 | 0.123 |
| kanvq_text_v2:text | 1319 | 3.0 | 2.2–4.0 | 184.1 | 0.0 | 184.1 | 113/22 | 6.09e-16 |
| vanilla_text:text | 1319 | 9.9 | 8.4–11.6 | 173.7 | 0.0 | 173.7 | — | — |
| sep_kan_plainhead_text:text | 1319 | 4.1 | 3.2–5.3 | 187.1 | 0.0 | 187.1 | 109/33 | 1.09e-10 |
| sep_swiglu_sphere_text:text | 1319 | 5.5 | 4.4–6.9 | 181.6 | 0.0 | 181.6 | 93/36 | 5.43e-07 |

### openmath — all  (reference: vanilla_text:text)

| run:mode | n | acc % | 95% CI | text | latent | steps | vs ref: b/c | McNemar p |
|---|---|---|---|---|---|---|---|---|
| kanvq_hybrid_v2:hybrid | 500 | 1.0 | 0.4–2.3 | 87.4 | 146.4 | 233.8 | 69/2 | 2.17e-18 |
| kanvq_hybrid_v2:text | 500 | 2.0 | 1.1–3.6 | 200.3 | 0.0 | 200.3 | 67/5 | 6.39e-15 |
| vanilla_hybrid_v2:hybrid | 500 | 3.4 | 2.1–5.4 | 49.4 | 140.4 | 189.8 | 62/7 | 4.1e-12 |
| vanilla_hybrid_v2:text | 500 | 11.8 | 9.3–14.9 | 178.8 | 0.0 | 178.8 | 44/31 | 0.165 |
| kanvq_text_v2:text | 500 | 3.4 | 2.1–5.4 | 179.2 | 0.0 | 179.2 | 63/8 | 1.03e-11 |
| vanilla_text:text | 500 | 14.4 | 11.6–17.7 | 175.5 | 0.0 | 175.5 | — | — |
| sep_kan_plainhead_text:text | 500 | 4.8 | 3.2–7.0 | 180.8 | 0.0 | 180.8 | 57/9 | 1.18e-09 |
| sep_swiglu_sphere_text:text | 500 | 8.4 | 6.3–11.2 | 180.0 | 0.0 | 180.0 | 51/21 | 0.000535 |