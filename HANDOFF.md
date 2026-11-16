# FractalGen-Histopatología 256 px — Guía de traspaso entre cuentas

> **Cómo usar este archivo:** pégalo (junto con el brief de tu rol, sección 6) al INICIO de cada chat nuevo con Claude. Así cualquier cuenta retoma el proyecto sin perder contexto. Actualiza las secciones 5 y 7 al terminar cada sesión.

## 1. Objetivo y requisitos (de Miguel)
- Generar imágenes de histopatología **de 256×256 px** con un modelo basado en el repo oficial **LTH14/fractalgen** (reusar sus modelos).
- Dataset **PanNuke**, solo órganos **Breast, Colon, Lung**, **solo imágenes buenas** (descartar las dañadas).
- GPU de Google Colab, **límite 15 GB**.
- Evaluar con FID, KID, IS y exactitud de un clasificador de órganos; interfaz para elegir órgano y generar.
- Versión previa (64 px, desde cero, 9 min) convergía a pérdida alta → se descarta como base.

## 2. Decisiones técnicas
| Decisión | Motivo |
|---|---|
| Fine-tuning de `fractalmar_base_in256` (186 M, preentrenado ImageNet-256) | desde cero con ~4 mil imágenes converge mal; el repo trae pesos |
| Config del repo: `guiding_pixel=True`, `num_conds=5`, `r_weight=5`, dropout 0.1, `label_drop=0.1` | es la del repo a 256 px; el notebook de 64 usaba otra |
| fp16 + GradScaler, grad checkpointing, micro-batch automático + acumulación (batch lógico 32) | caber en 15 GB (la T4 no tiene bf16) |
| `class_emb` 1000→3 inicializado con media de ImageNet + ruido | única capa que cambia de forma |
| LR 2e-5 (warmup 100, coseno a 2e-6), AdamW β=(0.9,0.95), wd 0.05, clip 3.0 | fine-tuning conservador |
| Aumentos D4 por imagen + muestreo balanceado por órgano | tejido sin orientación; Lung tiene menos datos |
| Filtro de calidad multi-criterio + galerías de revisión + `quality_report.csv` | auditable y ajustable |
| Métricas contra un **piso real-vs-real** | FID con pocas imágenes está sesgado; el piso da la referencia |
| Repo fijado en un commit (`fg_config.REPO_COMMIT`) | reproducibilidad |

## 3. Estructura (en Drive: `MyDrive/fractal_histopatologia_256/`)
```
code/            fg_config.py  fg_quality.py  fg_common.py  fg_eval.py     <- subir tal cual
notebooks/       01_datos  02_entrenamiento  03_evaluacion  04_interfaz   (.ipynb)
data/            X256.npy  y.npy  meta.json  split.npz                     <- salida de 01
pretrained/      fractalmar_base_in256.pth                                  <- se descarga solo
checkpoints/     last.pth (completo, reanuda)  best_fp16.pth  final_fp16.pth  clasificador_resnet18.pth
results/         quality_report.csv  history.json  config_gen.json  metricas_*.{json,csv}  generadas/  barrido_cfg/  interfaz/
```
**Puesta en marcha:** subir la carpeta `fractal_histopatologia_256` a `MyDrive`, abrir cada notebook en Colab (Entorno > GPU T4). Ajustar `DATASET_DIR` en `code/fg_config.py` si tu carpeta `.arrow` está en otra ruta.

## 4. Contratos entre módulos (no romper: permiten trabajar en paralelo)
| Archivo | Contenido | Produce → Consume |
|---|---|---|
| `data/X256.npy` | `uint8 [N,256,256,3]` solo imágenes buenas | 01 → 02, 03, 04 |
| `data/y.npy` | `int [N]`, índice de clase = orden de `cfg.ORGANS` | 01 → resto |
| `data/meta.json` | `{"classes": [...], "kept_row_ids": [...], ...}` | 01 → resto |
| `data/split.npz` | `train`, `val`: índices sobre `X256.npy` | 01 → resto |
| `checkpoints/best_fp16.pth` | `{"model": state_dict fp16, "classes", "size"}` | 02 → 03, 04 |
| `checkpoints/last.pth` | modelo + optimizador + scaler + `state` (reanudar) | 02 → 02 |
| `results/config_gen.json` | `{"cfg", "num_iter_list", "temperature"}` | 03 → 04 |
| `results/metricas_globales.json` | FID, FID_piso, KID, KID_piso, IS, exactitudes | 03 → 04 |
| PNG en `results/generadas/cfgX/<órgano>/NNNN.png` | imágenes 256×256 | 03 → 03 |
API de generación: `fg.generate_many(model, labels, bs, seed, cfg_scale=..., temperature=..., num_iter_list=...)`.

## 5. Estado del proyecto (ACTUALIZAR)
- [ ] 01 · Datos: dataset limpio guardado (anotar # buenas por órgano y umbrales finales)
- [ ] 02 · Benchmark hecho (anotar micro-batch, s/paso, ETA) y `TOTAL_STEPS` elegido
- [ ] 02 · Entrenamiento terminado (anotar paso final y mejor val loss)
- [ ] 03 · CFG elegido (anotar valor) y métricas calculadas
- [ ] 04 · Interfaz probada
- [ ] Informe final (figuras: `reales_vs_generadas.png`, curvas de `history.json`, tabla de métricas)

## 6. Reparto entre 4 cuentas (briefs para pegar)
Regla común: **pegar este archivo + solo los archivos de tu rol**; pedir cambios como *diffs* o funciones completas pequeñas; no tocar archivos de otro rol sin avisar; respetar los contratos de la sección 4.

**Cuenta A — Datos** · Archivos: `fg_config.py` (QUALITY), `fg_quality.py`, `01_datos.ipynb`.
> Eres responsable de que `data/` quede perfecto. Tarea: revisar los criterios de "imagen dañada" con las galerías, ajustar umbrales, documentar cuántas se descartan por razón y órgano. No cambies el formato de los archivos de la sección 4.

**Cuenta B — Entrenamiento** · Archivos: `fg_config.py` (modelo/entrenamiento), `fg_common.py`, `02_entrenamiento.ipynb`.
> Eres responsable del entrenamiento. Tarea: correr el benchmark, fijar `TOTAL_STEPS`, entrenar, vigilar val loss, resolver OOM/NaN. Entrega `best_fp16.pth`. Si cambias firmas en `fg_common.py`, avisa a C y D.

**Cuenta C — Evaluación** · Archivos: `fg_eval.py`, `03_evaluacion.ipynb`.
> Eres responsable de las métricas. Tarea: barrido de CFG, generar N por órgano, FID/KID/IS contra el piso, exactitud del clasificador. Entrega `config_gen.json` y `metricas_*`.

**Cuenta D — Interfaz e informe** · Archivos: `04_interfaz.ipynb`, este `HANDOFF.md`, figuras de `results/`.
> Eres responsable de la interfaz y el informe: probar la UI, mejorar la presentación, redactar el informe (metodología, decisiones y su porqué, resultados, limitaciones) con las métricas y figuras producidas por A–C.

## 7. Bitácora (añadir una línea por sesión: fecha · cuenta · qué se hizo · qué falta)
- (sin entradas aún)

## 8. Limitaciones conocidas (honestidad)
- El código fue escrito contra el repo oficial y validado en sintaxis y lógica, **pero no se ejecutó en una GPU**: el primer run puede requerir pequeños ajustes. Los tiempos (s/paso, s/imagen) se miden en el benchmark; no están estimados aquí.
- Los umbrales de calidad son heurísticos: la verificación visual con las galerías es parte del método.
- Con ~100 imágenes por órgano generadas, FID/IS son orientativos; KID y el piso real-vs-real dan mejor referencia.
- Las imágenes sintéticas son solo para investigación/educación, no para diagnóstico.

## 9. Solución de problemas rápida
OOM → `cfg.MICRO_BATCH=1` o `EFFECTIVE_BATCH=16` · NaN → `cfg.LR=1e-5` · descarga del preentrenado falla → subirlo a mano a `pretrained/` o `cfg.MODEL_SIZE="small"` · Colab se cae → reconectar y reejecutar 02 (reanuda de `last.pth`) · generación muy lenta → `cfg.NUM_ITER_LIST=(32,8,8)`.
