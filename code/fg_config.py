"""fg_config.py — ÚNICA fuente de verdad de rutas e hiperparámetros.

Todos los módulos y notebooks leen estos valores con `import fg_config as cfg`.
Para cambiar algo en una sesión sin tocar el archivo:  cfg.TOTAL_STEPS = 4000
(los módulos leen `cfg.X` en el momento de usarlo, así que el cambio se aplica).
"""

# ------------------------------------------------------------------ Rutas
DRIVE_ROOT  = "/content/drive/MyDrive"
PROJECT_DIR = f"{DRIVE_ROOT}/fractal_histopatologia_256"
DATASET_DIR = f"{DRIVE_ROOT}/Proyecto/DataSet_V0.1"      # <-- AJUSTA: carpeta con los .arrow de PanNuke
LOCAL_ARROW = "/content/pannuke_arrow"                   # copia local (leer de Drive es lento)

CODE_DIR = f"{PROJECT_DIR}/code"
DATA_DIR = f"{PROJECT_DIR}/data"
PRE_DIR  = f"{PROJECT_DIR}/pretrained"
CKPT_DIR = f"{PROJECT_DIR}/checkpoints"
RES_DIR  = f"{PROJECT_DIR}/results"

# Repo oficial (commit fijado para que el proyecto sea reproducible)
REPO_DIR    = "/content/fractalgen"
REPO_URL    = "https://github.com/LTH14/fractalgen.git"
REPO_COMMIT = "c7d099043dd987ae7742a7f8c8f1cab71023ba0e"

# ------------------------------------------------------------------ Datos
ORGANS        = ["Breast", "Colon", "Lung"]
IMG_SIZE      = 256            # PanNuke ya es 256x256 nativo: no se reescala nada
SEED          = 0
TEST_FRACTION = 0.10           # validación (split estratificado por órgano)

# Criterios de "imagen dañada" (se calculan sobre la imagen 256x256 en RGB 0-255).
# Revisa las galerías del notebook 01 y ajusta si hay falsos positivos / negativos.
QUALITY = dict(
    min_mean=100,     # brillo medio (gris) mínimo  -> "oscura"   (tu hallazgo previo con Colon)
    min_std=15,       # contraste mínimo (desv. del gris) -> "sin_contraste"
    max_black=0.10,   # fracción máx. de píxeles casi negros (<15) -> "zonas_negras"
    max_white=0.60,   # fracción máx. de píxeles casi blancos (>235) -> "mayormente_blanca"
    min_sat=0.06,     # saturación media mínima -> "casi_gris"
    min_stain=0.0,    # (R+B)/2 - G medio mínimo; H&E es rosa/morado -> "tono_no_HE"
    blur_rel=0.25,    # borrosa si nitidez < 25 % de la mediana del órgano ...
    blur_pct=2.0,     # ... y además está en el 2 % más bajo del órgano
)
MANUAL_BLACKLIST  = []         # lista de row_id (columna del quality_report.csv) a excluir a mano
REMOVE_DUPLICATES = True

# ------------------------------------------------------------------ Modelo
# "base"  : FractalMAR-Base 186M, preentrenado en ImageNet-256 (fine-tuning)  <- recomendado
# "small" : ~25M, desde cero (solo si el preentrenado no se puede descargar)
MODEL_SIZE         = "base"
NUM_CONDS          = 5         # el forward de FractalGen lo fija en 5
GUIDING_PIXEL      = True      # el repo lo usa en 256 px
R_WEIGHT           = 5.0       # valor por defecto del repo (peso del canal rojo en la pérdida)
ATTN_DROPOUT       = 0.1
PROJ_DROPOUT       = 0.1
LABEL_DROP_PROB    = 0.1       # necesario para classifier-free guidance
GRAD_CHECKPOINTING = True      # ahorra memoria de activaciones (clave en 15 GB)

# ------------------------------------------------------------------ Entrenamiento
EFFECTIVE_BATCH   = 32         # batch lógico (se logra con micro-batch x acumulación)
MICRO_BATCH       = None       # None = se detecta solo el mayor que cabe en la GPU
LR                = 2e-5       # fine-tuning: bajo
MIN_LR            = 2e-6
WARMUP_STEPS      = 100
WEIGHT_DECAY      = 0.05
BETAS             = (0.9, 0.95)
GRAD_CLIP         = 3.0
TOTAL_STEPS       = 3000       # pasos de optimizador; ajusta tras ver el ETA del benchmark
SESSION_MINUTES   = 150        # corta solo antes del límite de Colab; reejecuta para continuar
SAVE_EVERY_MIN    = 20         # checkpoint completo en Drive
VAL_EVERY         = 250
VAL_IMAGES        = 64
BALANCED_SAMPLING = True       # cada órgano aparece con la misma frecuencia

# ------------------------------------------------------------------ Generación
NUM_ITER_LIST    = (64, 16, 16)   # calidad. Rápido: (32, 8, 8)
CFG              = 3.0            # se calibra en el notebook 03 (barrido de CFG)
CFG_SCHEDULE     = "linear"
TEMPERATURE      = 1.0
FILTER_THRESHOLD = 1e-4
