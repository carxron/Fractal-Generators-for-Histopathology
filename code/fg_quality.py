"""fg_quality.py — detección de imágenes dañadas de PanNuke (sin torch).

Flujo: image_metrics() por imagen -> DataFrame -> flag_quality() -> galerías para
revisar a ojo -> se guardan solo las imágenes con keep=True.
"""
import hashlib
import numpy as np
import pandas as pd
from scipy import ndimage

import fg_config as cfg

_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)
METRIC_COLS = ["mean_gray", "std_gray", "frac_black", "frac_white", "lap_var", "sat", "stain"]


def image_metrics(img):
    """img: HxWx3 uint8. Devuelve un dict de métricas de calidad."""
    a = img.astype(np.float32)
    gray = a @ _GRAY
    mx, mn = a.max(2), a.min(2)
    return dict(
        mean_gray=float(gray.mean()),                    # brillo
        std_gray=float(gray.std()),                      # contraste
        frac_black=float((gray < 15).mean()),            # zonas negras / relleno
        frac_white=float((gray > 235).mean()),           # vidrio vacío
        lap_var=float(ndimage.laplace(gray).var()),      # nitidez (varianza del Laplaciano)
        sat=float(((mx - mn) / np.maximum(mx, 1)).mean()),   # saturación media
        stain=float(((a[..., 0] + a[..., 2]) / 2 - a[..., 1]).mean()),  # rosa/morado (H&E) > 0
    )


def md5_of(img):
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def flag_quality(df, q=None, blacklist=None, remove_duplicates=None):
    """df necesita: row_id, organ, decode_error(bool), md5 y METRIC_COLS.
    Devuelve copia con 'reasons' (texto separado por ;) y 'keep' (bool)."""
    q = {**cfg.QUALITY, **(q or {})}
    blacklist = cfg.MANUAL_BLACKLIST if blacklist is None else blacklist
    remove_duplicates = cfg.REMOVE_DUPLICATES if remove_duplicates is None else remove_duplicates

    df = df.copy().reset_index(drop=True)
    n = len(df)
    R = [[] for _ in range(n)]

    def add(mask, name):
        for i in np.flatnonzero(np.asarray(mask)):
            R[i].append(name)

    err = df["decode_error"].to_numpy(bool)
    ok = ~err
    g = lambda c: df[c].to_numpy(float)
    organ = df["organ"].to_numpy()

    with np.errstate(invalid="ignore"):
        add(err, "corrupta")
        add(ok & (g("mean_gray") < q["min_mean"]), "oscura")
        add(ok & (g("std_gray") < q["min_std"]), "sin_contraste")
        add(ok & (g("frac_black") > q["max_black"]), "zonas_negras")
        add(ok & (g("frac_white") > q["max_white"]), "mayormente_blanca")
        add(ok & (g("sat") < q["min_sat"]), "casi_gris")
        add(ok & (g("stain") < q["min_stain"]), "tono_no_HE")

        # borrosa: relativa a cada órgano (cada tejido tiene su propia nitidez típica)
        lap = g("lap_var")
        blur = np.zeros(n, bool)
        for o in np.unique(organ):
            m = ok & (organ == o)
            if m.sum() < 20:
                continue
            med = np.nanmedian(lap[m])
            pct = np.nanpercentile(lap[m], q["blur_pct"])
            blur |= m & (lap <= min(q["blur_rel"] * med, pct))
        add(blur, "borrosa")

    if remove_duplicates:
        add(ok & df.duplicated("md5", keep="first").to_numpy(), "duplicada")
    if blacklist:
        add(df["row_id"].isin(list(blacklist)).to_numpy(), "manual")

    df["reasons"] = [";".join(r) for r in R]
    df["keep"] = [len(r) == 0 for r in R]
    return df


def reasons_table(df):
    """Tabla razón x órgano con cuántas imágenes se descartan (una imagen puede tener varias razones)."""
    bad = df[~df["keep"]].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["reason"] = bad["reasons"].str.split(";")
    ex = bad.explode("reason").reset_index(drop=True)
    t = pd.crosstab(ex["reason"], ex["organ"])
    t["total"] = t.sum(axis=1)
    return t.sort_values("total", ascending=False)


def summary(df):
    """Resumen por órgano: total, descartadas, conservadas."""
    g = df.groupby("organ")["keep"].agg(total="size", conservadas="sum")
    g["descartadas"] = g["total"] - g["conservadas"]
    g["% descartado"] = (100 * g["descartadas"] / g["total"]).round(1)
    return g


# ------------------------------------------------------------------ galerías (para revisar a ojo)
def _grid(X, df, ids, title, ncols=8, size=1.7, show_cols=("mean_gray", "std_gray", "lap_var")):
    import matplotlib.pyplot as plt
    ids = list(ids)
    if not ids:
        print(f"[{title}] sin imágenes"); return
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * (size + 0.45)))
    axs = np.atleast_1d(axs).ravel()
    for a in axs:
        a.axis("off")
    for a, i in zip(axs, ids):
        a.imshow(X[i])
        r = df.iloc[i]
        a.set_title(f"{r['organ']} #{int(r['row_id'])}\nbr{r['mean_gray']:.0f} ct{r['std_gray']:.0f} nt{r['lap_var']:.0f}",
                    fontsize=6.5)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def show_rejected(X, df, reason, n=16, seed=0):
    """Muestra n imágenes descartadas por una razón concreta."""
    m = df["reasons"].str.contains(reason) & ~df["keep"]
    ids = np.flatnonzero(m.to_numpy())
    rng = np.random.RandomState(seed)
    sel = rng.choice(ids, min(n, len(ids)), replace=False) if len(ids) else []
    _grid(X, df, sel, f"Descartadas por '{reason}' ({len(ids)} en total)")


def show_kept(X, df, n_per_organ=8, seed=0):
    """Muestra imágenes conservadas al azar, por órgano (para confirmar que son buenas)."""
    rng = np.random.RandomState(seed)
    for o in sorted(df["organ"].unique()):
        ids = np.flatnonzero(((df["organ"] == o) & df["keep"]).to_numpy())
        sel = rng.choice(ids, min(n_per_organ, len(ids)), replace=False)
        _grid(X, df, sel, f"Conservadas — {o}", ncols=n_per_organ)


def show_borderline(X, df, n=16, seed=0):
    """Conservadas pero con brillo/contraste cercano al umbral: ayuda a decidir si endurecer el filtro."""
    q = cfg.QUALITY
    d = df[df["keep"]].copy()
    d["margen"] = np.minimum((d["mean_gray"] - q["min_mean"]) / q["min_mean"],
                             (d["std_gray"] - q["min_std"]) / q["min_std"])
    ids = d.sort_values("margen").head(n).index.to_numpy()
    _grid(X, df, ids, "Conservadas más cercanas al umbral (revisa si las dejarías)")
