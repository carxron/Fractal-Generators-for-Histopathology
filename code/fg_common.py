"""fg_common.py — núcleo del proyecto: datos, modelo FractalGen, entrenamiento y generación.

Se apoya en el repo oficial LTH14/fractalgen (clonado en cfg.REPO_DIR). Nada de aquí
reimplementa el modelo: solo lo construye, lo ajusta (fine-tuning) y lo muestrea.
"""
import os, sys, json, math, time, random, shutil, subprocess
import numpy as np
import torch

import fg_config as cfg

PIX_MEAN = (0.485, 0.456, 0.406)   # mismas constantes que usa PixelLoss del repo
PIX_STD = (0.229, 0.224, 0.225)


# =============================================================== entorno
def setup_dirs():
    for d in (cfg.PROJECT_DIR, cfg.CODE_DIR, cfg.DATA_DIR, cfg.PRE_DIR, cfg.CKPT_DIR, cfg.RES_DIR):
        os.makedirs(d, exist_ok=True)


def mount_drive():
    if not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    setup_dirs()


def setup_repo():
    """Clona el repo oficial en el commit fijado y lo deja en sys.path (y como directorio actual)."""
    if not os.path.exists(cfg.REPO_DIR):
        subprocess.run(["git", "clone", "-q", cfg.REPO_URL, cfg.REPO_DIR], check=True)
        subprocess.run(["git", "-C", cfg.REPO_DIR, "checkout", "-q", cfg.REPO_COMMIT], check=True)
    os.chdir(cfg.REPO_DIR)           # util/download.py guarda los pesos relativo al cwd
    if cfg.REPO_DIR not in sys.path:
        sys.path.insert(0, cfg.REPO_DIR)


def seed_everything(seed=None):
    seed = cfg.SEED if seed is None else seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_info():
    assert torch.cuda.is_available(), "Activa la GPU: Entorno de ejecución > Cambiar tipo de entorno > GPU"
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | {p.total_memory / 2**30:.1f} GB")
    return p.total_memory / 2**30


# =============================================================== datos
def normalize(x01):
    m = torch.tensor(PIX_MEAN, device=x01.device).view(1, 3, 1, 1)
    s = torch.tensor(PIX_STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - m) / s


def denormalize(x):
    m = torch.tensor(PIX_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(PIX_STD, device=x.device).view(1, 3, 1, 1)
    return (x * s + m).clamp(0, 1)


def augment_d4(x):
    """Flips + rotaciones de 90° independientes por imagen (el tejido no tiene orientación)."""
    B = x.size(0)
    k = torch.randint(0, 4, (B,), device=x.device)
    out = x.clone()
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            out[m] = torch.rot90(x[m], r, (2, 3))
    f = torch.rand(B, device=x.device) < 0.5
    return torch.where(f.view(-1, 1, 1, 1), out.flip(3), out)


def load_clean_data():
    """Carga lo que dejó el notebook 01. Devuelve X(uint8 NHWC), y, classes, train_idx, val_idx."""
    X = np.load(f"{cfg.DATA_DIR}/X256.npy")
    y = np.load(f"{cfg.DATA_DIR}/y.npy")
    classes = json.load(open(f"{cfg.DATA_DIR}/meta.json"))["classes"]
    sp = np.load(f"{cfg.DATA_DIR}/split.npz")
    return X, y, classes, sp["train"], sp["val"]


class BatchSampler:
    """Muestrea batches desde RAM (CPU) y los sube a la GPU. Balanceado por órgano si balanced=True."""

    def __init__(self, X, y, idx, device="cuda", balanced=None, seed=0):
        balanced = cfg.BALANCED_SAMPLING if balanced is None else balanced
        self.X, self.y, self.idx, self.device = X, y, np.asarray(idx), device
        yy = y[self.idx]
        if balanced:
            w = 1.0 / np.bincount(yy)[yy]
            self.p = w / w.sum()
        else:
            self.p = None
        self.rng = np.random.RandomState(seed)

    def get(self, bs, augment=True):
        sel = self.rng.choice(self.idx, size=bs, replace=True, p=self.p)
        x = torch.from_numpy(self.X[sel]).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        if augment:
            x = augment_d4(x)
        return normalize(x), torch.from_numpy(self.y[sel]).long().to(self.device)


# =============================================================== modelo
def build_model(num_classes, size=None, grad_checkpointing=None):
    """FractalMAR de 4 niveles (256 -> 16 -> 4 -> 1) del repo oficial, con class_num=num_classes."""
    from models.fractalgen import FractalGen, fractalmar_base_in256
    size = size or cfg.MODEL_SIZE
    kw = dict(label_drop_prob=cfg.LABEL_DROP_PROB, class_num=num_classes,
              attn_dropout=cfg.ATTN_DROPOUT, proj_dropout=cfg.PROJ_DROPOUT,
              guiding_pixel=cfg.GUIDING_PIXEL, num_conds=cfg.NUM_CONDS, r_weight=cfg.R_WEIGHT,
              grad_checkpointing=cfg.GRAD_CHECKPOINTING if grad_checkpointing is None else grad_checkpointing)
    if size == "base":
        return fractalmar_base_in256(**kw)
    assert size == "small", "MODEL_SIZE debe ser 'base' o 'small'"
    return FractalGen(img_size_list=(256, 16, 4, 1), embed_dim_list=(384, 192, 96, 64),
                      num_blocks_list=(12, 4, 2, 1), num_heads_list=(6, 3, 3, 4),
                      generator_type_list=("mar", "mar", "mar", "ar"), fractal_level=0, **kw)


def get_pretrained_path():
    """Descarga (una sola vez) FractalMAR-Base ImageNet-256 (~712 MB) y la guarda en Drive."""
    dst = f"{cfg.PRE_DIR}/fractalmar_base_in256.pth"
    if os.path.exists(dst) and os.path.getsize(dst) > 600e6:
        return dst
    from util.download import download_pretrained_fractalmar_base_in256
    download_pretrained_fractalmar_base_in256()
    src = f"{cfg.REPO_DIR}/pretrained_models/fractalmar_base_in256/checkpoint-last.pth"
    if not os.path.exists(src) or os.path.getsize(src) < 600e6:
        raise RuntimeError(
            "La descarga del preentrenado falló (Dropbox suele limitar descargas). Descarga a mano el archivo "
            "'checkpoint-last.pth' del enlace FractalMAR-Base (IN256) del README del repo, súbelo a "
            f"{dst} y vuelve a ejecutar. Alternativa: cfg.MODEL_SIZE = 'small' (entrena desde cero).")
    shutil.copy(src, dst)
    return dst


def load_pretrained(model, path):
    """Carga los pesos de ImageNet. Solo cambia class_emb (1000 clases -> N): se inicia con la media
    de los embeddings de ImageNet + ruido pequeño, para que las clases arranquen distintas entre sí."""
    ck = torch.load(path, map_location="cpu", weights_only=False)   # checkpoint oficial del repo (fuente confiable)
    sd = ck["model"]
    old = sd.pop("class_emb.weight")
    n = model.class_emb.num_embeddings
    sd["class_emb.weight"] = old.mean(0, keepdim=True).repeat(n, 1) + 0.02 * torch.randn(n, old.size(1))
    model.load_state_dict(sd, strict=True)
    print(f"Preentrenado cargado ({len(sd)} tensores). class_emb: {tuple(old.shape)} -> {(n, old.size(1))}")


def n_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def param_groups(model, wd):
    """Sin weight decay en sesgos / normas (igual que util.misc.add_weight_decay del repo)."""
    dec, nodec = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodec if p.ndim == 1 or name.endswith(".bias") else dec).append(p)
    return [{"params": nodec, "weight_decay": 0.0}, {"params": dec, "weight_decay": wd}]


# =============================================================== entrenamiento
def lr_at(step):
    if step < cfg.WARMUP_STEPS:
        return cfg.LR * (step + 1) / cfg.WARMUP_STEPS
    p = (step - cfg.WARMUP_STEPS) / max(1, cfg.TOTAL_STEPS - cfg.WARMUP_STEPS)
    return cfg.MIN_LR + (cfg.LR - cfg.MIN_LR) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def find_micro_batch(model, sampler, total_gb, headroom_gb=2.4, device="cuda"):
    """Prueba micro-batches de mayor a menor; elige el mayor cuyo pico de memoria deja
    `headroom_gb` libres (para el estado de AdamW, ~1.5 GB, y fragmentación). Mide s/paso."""
    eff = cfg.EFFECTIVE_BATCH
    cands = [c for c in (32, 16, 8, 4, 2, 1) if c <= eff and eff % c == 0]
    if cfg.MICRO_BATCH:
        cands = [cfg.MICRO_BATCH]
    model.train()
    for mb in cands:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            for it in range(3):                      # 1 de calentamiento + 2 medidos
                if it == 1:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                x, y = sampler.get(mb)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(x, y)
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            sec = (time.perf_counter() - t0) / 2
            peak = torch.cuda.max_memory_allocated() / 2**30
            if peak + headroom_gb > total_gb:
                print(f"micro-batch {mb}: pico {peak:.1f} GB, sin margen para AdamW -> pruebo menor")
                continue
            print(f"micro-batch {mb}: pico {peak:.1f} GB | {sec:.2f} s por micro-batch  OK")
            return mb, sec, peak
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not _is_oom(e):
                raise
            model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print(f"micro-batch {mb}: sin memoria -> pruebo menor")
    raise RuntimeError("Ni con micro-batch 1 cabe. Prueba cfg.MODEL_SIZE='small'.")


@torch.no_grad()
def val_loss(model, X, y, val_idx, n=None, bs=4, device="cuda"):
    """NLL por sub-píxel en validación (modo eval = incondicional, máscara uniforme, semilla fija).
    Sirve para COMPARAR pasos/configuraciones, no como valor absoluto."""
    n = n or cfg.VAL_IMAGES
    rng = np.random.RandomState(123)
    ids = rng.choice(val_idx, min(n, len(val_idx)), replace=False)
    st = (torch.get_rng_state(), torch.cuda.get_rng_state(), np.random.get_state())
    torch.manual_seed(123); torch.cuda.manual_seed(123); np.random.seed(123)
    model.eval()
    tot = 0.0
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        x = normalize(torch.from_numpy(X[b]).to(device).permute(0, 3, 1, 2).float() / 255.0)
        lab = torch.from_numpy(y[b]).long().to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            tot += model(x, lab).item() * len(b)
    model.train()
    torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1]); np.random.set_state(st[2])
    return tot / len(ids)


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_train_state(model, opt, scaler, state, classes):
    """Checkpoint completo (modelo + AdamW + scaler + paso + historial) para REANUDAR. ~2.2 GB."""
    _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                  "state": state, "classes": classes, "size": cfg.MODEL_SIZE}, f"{cfg.CKPT_DIR}/last.pth")
    json.dump(state["history"], open(f"{cfg.RES_DIR}/history.json", "w"))


def save_model_only(model, classes, name):
    """Solo pesos en fp16 (~0.4 GB): lo que usan evaluación e interfaz."""
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    _atomic_save({"model": sd, "classes": classes, "size": cfg.MODEL_SIZE}, f"{cfg.CKPT_DIR}/{name}")


def load_for_inference(name="best_fp16.pth", device="cuda"):
    ck = torch.load(f"{cfg.CKPT_DIR}/{name}", map_location="cpu", weights_only=False)
    classes = ck["classes"]
    model = build_model(len(classes), size=ck.get("size"), grad_checkpointing=False)
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["model"].items()})
    return model.to(device).eval(), classes


def new_state():
    return {"step": 0, "best_val": float("inf"),
            "history": {"step": [], "loss": [], "val_step": [], "val_loss": []}}


def has_checkpoint():
    return os.path.exists(f"{cfg.CKPT_DIR}/last.pth")


def try_resume(model, opt, scaler):
    """Si existe last.pth en Drive, restaura modelo + AdamW + scaler + paso; si no, estado nuevo."""
    p = f"{cfg.CKPT_DIR}/last.pth"
    if not os.path.exists(p):
        return new_state()
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
    print(f"Reanudado desde el paso {ck['state']['step']} (mejor val {ck['state']['best_val']:.4f})")
    st = ck["state"]; del ck
    return st


def train_loop(model, opt, scaler, sampler, data, micro, accum, state, device="cuda"):
    """Bucle reanudable. Se detiene solo al llegar a TOTAL_STEPS o a SESSION_MINUTES;
    guarda en Drive periódicamente y al terminar. Volver a ejecutarlo continúa donde quedó."""
    X, y, classes, val_idx = data
    hist = state["history"]
    t_start = t_save = time.time()
    run, skipped = [], 0
    model.train()
    try:
        while state["step"] < cfg.TOTAL_STEPS:
            if (time.time() - t_start) / 60 >= cfg.SESSION_MINUTES:
                print(f"Fin de sesión ({cfg.SESSION_MINUTES} min). Vuelve a ejecutar esta celda para continuar.")
                break
            step = state["step"]
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for _ in range(accum):
                x, lab = sampler.get(micro)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(x, lab) / accum
                scaler.scale(loss).backward()
                tot += loss.item()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(opt)             # si hay inf/NaN en los gradientes, el scaler salta el paso solo
            scaler.update()
            state["step"] += 1
            if math.isfinite(tot):
                run.append(tot); hist["step"].append(state["step"]); hist["loss"].append(tot)
            else:
                skipped += 1

            s = state["step"]
            if s % 25 == 0:
                el = (time.time() - t_start) / 60
                print(f"paso {s}/{cfg.TOTAL_STEPS} | loss {np.mean(run[-25:]):.4f} | lr {lr:.1e} | "
                      f"|grad| {float(gn):.2f} | omitidos {skipped} | {el:.1f} min")
            if s % cfg.VAL_EVERY == 0 or s == cfg.TOTAL_STEPS:
                v = val_loss(model, X, y, val_idx, device=device)
                hist["val_step"].append(s); hist["val_loss"].append(v)
                mejor = v < state["best_val"]
                print(f"   >>> val loss {v:.4f} (mejor previo {state['best_val']:.4f})" + ("  * nuevo mejor" if mejor else ""))
                if mejor:
                    state["best_val"] = v
                    save_model_only(model, classes, "best_fp16.pth")
            if (time.time() - t_save) / 60 >= cfg.SAVE_EVERY_MIN:
                save_train_state(model, opt, scaler, state, classes); t_save = time.time()
                print("   checkpoint guardado en Drive")
    except KeyboardInterrupt:
        print("Interrumpido manualmente: guardando antes de salir...")
    save_train_state(model, opt, scaler, state, classes)
    save_model_only(model, classes, "final_fp16.pth")
    print(f"Guardado. Paso actual: {state['step']}/{cfg.TOTAL_STEPS} | mejor val: {state['best_val']:.4f}")


def plot_history(hist):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    if hist["loss"]:
        k = min(25, len(hist["loss"]))
        sm = np.convolve(hist["loss"], np.ones(k) / k, mode="valid")
        ax[0].plot(hist["step"][k - 1:], sm); ax[0].set_title("Loss de entrenamiento (suavizada)")
        ax[0].set_xlabel("paso")
    if hist["val_loss"]:
        ax[1].plot(hist["val_step"], hist["val_loss"], "o-"); ax[1].set_title("Loss de validación")
        ax[1].set_xlabel("paso")
    plt.tight_layout(); plt.show()


# =============================================================== generación
@torch.no_grad()
def generate(model, labels, cfg_scale=None, temperature=None, num_iter_list=None, cfg_schedule=None,
             filter_threshold=None, device="cuda"):
    """Muestreo igual que engine_fractalgen.evaluate del repo. labels: índices de clase.
    Devuelve [N,3,256,256] en [0,1] (CPU)."""
    cfg_scale = cfg.CFG if cfg_scale is None else cfg_scale
    temperature = cfg.TEMPERATURE if temperature is None else temperature
    num_iter_list = cfg.NUM_ITER_LIST if num_iter_list is None else num_iter_list
    cfg_schedule = cfg.CFG_SCHEDULE if cfg_schedule is None else cfg_schedule
    filter_threshold = cfg.FILTER_THRESHOLD if filter_threshold is None else filter_threshold
    model.eval()
    labels = torch.as_tensor(labels).long().to(device)
    emb = model.class_emb(labels)
    if cfg_scale != 1.0:                       # CFG: se concatena el embedding "nulo"
        emb = torch.cat([emb, model.fake_latent.repeat(labels.size(0), 1)], dim=0)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model.sample(cond_list=[emb for _ in range(cfg.NUM_CONDS)], num_iter_list=list(num_iter_list),
                           cfg=cfg_scale, cfg_schedule=cfg_schedule, temperature=temperature,
                           filter_threshold=filter_threshold, fractal_level=0)
    return denormalize(out.float()).cpu()


def generate_many(model, labels, bs=4, seed=None, **kw):
    """Genera por tandas; si hay OOM reduce la tanda automáticamente."""
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed % (2**32 - 1))
    labels = list(labels)
    out, i = [], 0
    while i < len(labels):
        try:
            out.append(generate(model, labels[i:i + bs], **kw)); i += bs
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not _is_oom(e) or bs == 1:
                raise
            torch.cuda.empty_cache(); bs = max(1, bs // 2)
            print(f"OOM al generar -> tanda de {bs}")
    return torch.cat(out)


def to_pil(img01):
    from PIL import Image
    return Image.fromarray((img01.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))


def generate_to_disk(model, classes, n_per_class, out_dir, bs=4, seed_base=0, **kw):
    """Genera n_per_class PNG por órgano en out_dir/<órgano>/. Reanudable: salta los que ya existen."""
    t0, made = time.time(), 0
    for ci, organ in enumerate(classes):
        d = f"{out_dir}/{organ}"
        os.makedirs(d, exist_ok=True)
        i = len([f for f in os.listdir(d) if f.endswith(".png")])
        while i < n_per_class:
            b = min(bs, n_per_class - i)
            imgs = generate_many(model, [ci] * b, bs=b, seed=seed_base + 1000 * ci + i, **kw)
            for k, im in enumerate(imgs):
                to_pil(im).save(f"{d}/{i + k:04d}.png")
            i += b; made += b
            print(f"{organ}: {i}/{n_per_class} | {(time.time() - t0) / made:.1f} s/img")
    return out_dir
