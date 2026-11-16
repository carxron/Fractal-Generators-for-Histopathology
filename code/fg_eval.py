"""fg_eval.py — métricas de evaluación: FID / KID / IS + clasificador de órganos (ResNet18).

Las métricas se calculan desde carpetas de PNG, así que generar y evaluar son pasos
independientes (y reanudables si Colab se desconecta).
"""
import os, glob
import numpy as np
import torch
import torch.nn as nn

import fg_config as cfg
import fg_common as fg


# =============================================================== utilidades
def load_png_folder(folder, max_n=None):
    files = sorted(glob.glob(f"{folder}/*.png"))[:max_n]
    from PIL import Image
    arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()          # uint8 NCHW


def _feed(metric, imgs_u8, device, bs=32, **kw):
    for i in range(0, len(imgs_u8), bs):
        metric.update(imgs_u8[i:i + bs].to(device), **kw)


def real_sample(X, y, idx, n_per_class, n_classes, seed=0):
    """uint8 NCHW de imágenes reales (n_per_class por órgano) tomadas de los índices idx."""
    rng = np.random.RandomState(seed)
    out = {}
    for c in range(n_classes):
        ids = idx[y[idx] == c]
        ids = rng.choice(ids, min(n_per_class, len(ids)), replace=False)
        out[c] = torch.from_numpy(X[ids]).permute(0, 3, 1, 2).contiguous()
    return out


# =============================================================== FID / KID / IS
def distribution_metrics(real_u8, gen_u8, device="cuda", with_is=True):
    """FID (↓), KID (↓, menos sesgada con pocas imágenes) e IS (↑) entre dos conjuntos uint8 NCHW."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    res = {}
    fid = FrechetInceptionDistance(feature=2048).to(device)
    _feed(fid, real_u8, device, real=True); _feed(fid, gen_u8, device, real=False)
    res["FID"] = fid.compute().item(); del fid
    sub = int(min(50, len(real_u8), len(gen_u8)))
    kid = KernelInceptionDistance(subset_size=sub).to(device)
    _feed(kid, real_u8, device, real=True); _feed(kid, gen_u8, device, real=False)
    m, s = kid.compute(); res["KID"], res["KID_std"] = m.item(), s.item(); del kid
    if with_is:
        isc = InceptionScore().to(device); _feed(isc, gen_u8, device)
        res["IS"] = isc.compute()[0].item(); del isc
    torch.cuda.empty_cache()
    return res


# =============================================================== clasificador de órganos
def train_classifier(X, y, train_idx, val_idx, classes, epochs=5, bs=64, device="cuda", force=False):
    """ResNet18 (ImageNet) ajustada para reconocer el órgano en imágenes REALES de 256 px.
    Se guarda en Drive; si ya existe se reutiliza."""
    import torchvision
    path = f"{cfg.CKPT_DIR}/clasificador_resnet18.pth"
    clf = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    clf.fc = nn.Linear(clf.fc.in_features, len(classes))
    clf = clf.to(device)
    if os.path.exists(path) and not force:
        clf.load_state_dict(torch.load(path, map_location=device)); clf.eval()
        print("Clasificador cargado de Drive"); return clf
    sampler = fg.BatchSampler(X, y, train_idx, device=device, balanced=True, seed=1)
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    steps = max(1, len(train_idx) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * steps)
    for ep in range(epochs):
        clf.train(); tot = 0.0
        for _ in range(steps):
            x, lab = sampler.get(bs)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = nn.functional.cross_entropy(clf(x), lab)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        print(f"clasificador época {ep + 1}/{epochs}: loss {tot / steps:.3f}")
    clf.eval(); torch.save(clf.state_dict(), path)
    return clf


@torch.no_grad()
def clf_probs(clf, imgs, device="cuda", bs=64):
    """imgs: float [0,1] NCHW o uint8 NCHW. Devuelve probabilidades [N, n_clases] (CPU)."""
    out = []
    for i in range(0, len(imgs), bs):
        x = imgs[i:i + bs].to(device)
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        out.append(clf(fg.normalize(x)).softmax(1).cpu())
    return torch.cat(out)


def real_accuracy(clf, X, y, val_idx, device="cuda"):
    x = torch.from_numpy(X[val_idx]).permute(0, 3, 1, 2)
    return (clf_probs(clf, x, device).argmax(1).numpy() == y[val_idx]).mean().item()


def gen_accuracy(clf, gen_by_class, classes, device="cuda"):
    """Exactitud y probabilidad media del órgano pedido sobre las imágenes generadas (dict organo -> uint8 NCHW)."""
    acc, prob = {}, {}
    for ci, o in enumerate(classes):
        p = clf_probs(clf, gen_by_class[o], device)
        acc[o] = (p.argmax(1) == ci).float().mean().item()
        prob[o] = p[:, ci].mean().item()
    return acc, prob


def show_grid(imgs01, titles, ncols, path=None, size=2.0):
    import matplotlib.pyplot as plt
    n = len(imgs01); nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.1))
    axs = np.atleast_1d(axs).ravel()
    for a in axs:
        a.axis("off")
    for a, im, t in zip(axs, imgs01, titles):
        a.imshow(im.permute(1, 2, 0).numpy()); a.set_title(t, fontsize=8)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
