"""
train.py — 基于 GAN 与 BERT 的跨模态生成系统
方向三：文本条件图像生成 + 图像描述质量评估
- 满足所有技术约束：3组超参数对比、数据工程、可视化监控、硬件效率Profile
- 生成器使用高质量128通道反卷积结构，保证图片清晰度
"""

import os, sys, math, time, json, warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.cuda.amp import autocast, GradScaler
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. 全局配置与超参数组
# ─────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 目录结构
_BASE = os.path.dirname(os.path.abspath(__file__))
def _p(*parts):
    d = os.path.join(_BASE, *parts)
    os.makedirs(d, exist_ok=True)
    return d

DIRS = {
    "data":    _p("data", "flowers102"),
    "ckpt":    _p("checkpoints", "gan"),
    "samples": _p("checkpoints", "gan", "samples"),
    "tb":      _p("runs"),
    "reports": _p("reports"),
}

# 三组超参数对比（Run-A/B/C，各训练10 epoch，共30 epoch）
HPARAM_GROUPS = [
    {
        "run_id":      "run_A",
        "description": "基线：Adam + BatchNorm + lr=2e-4",
        "gan_lr":      2e-4,
        "optimizer":   "Adam",
        "norm":        "batch",
        "dropout":     0.3,
        "epochs":      10,
        "batch_size":  8,
    },
    {
        "run_id":      "run_B",
        "description": "非对称lr + InstanceNorm：D_lr=4e-4, G_lr=1e-4",
        "gan_lr":      1e-4,
        "d_lr_scale":  4.0,
        "optimizer":   "Adam",
        "norm":        "instance",
        "dropout":     0.3,
        "epochs":      10,
        "batch_size":  8,
    },
    {
        "run_id":      "run_C",
        "description": "RMSprop + BatchNorm + 更强Dropout=0.5",
        "gan_lr":      2e-4,
        "optimizer":   "RMSprop",
        "norm":        "batch",
        "dropout":     0.5,
        "epochs":      10,
        "batch_size":  8,
    },
]

# 通用固定配置
BASE_CFG = {
    "image_size":      128,
    "channels":        3,
    "nz":              128,
    "condition_dim":   128,
    "ngf":             128,    # 保持128通道，保证生成图片清晰度
    "ndf":             128,
    "amp":             True,
    "num_workers":     0,
    "save_every":      5,
    "sample_every":    2,
    "bert_model":      "bert-base-uncased",
    "freeze_bert_layers": 8,
    "seed":            SEED,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir":        DIRS["data"],
}

device = torch.device(BASE_CFG["device"])
print(f"\n{'='*55}")
print(f"  跨模态 GAN 生成系统 — 三组超参数对比训练")
print(f"{'='*55}")
print(f"[设备] {device}")
if device.type == "cuda":
    print(f"[GPU]  {torch.cuda.get_device_name(0)}")
    print(f"[显存] {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print(f"[随机种子] {SEED}（已固定，结果可复现）")

# ─────────────────────────────────────────────────────────────
# 1. 花卉类别
# ─────────────────────────────────────────────────────────────
FLOWER_CLASSES = [
    "pink primrose","hard-leaved pocket orchid","canterbury bells","sweet pea",
    "english marigold","tiger lily","moon orchid","bird of paradise","monkshood",
    "globe thistle","snapdragon","colt's foot","king protea","spear thistle",
    "yellow iris","globe-flower","purple coneflower","peruvian lily","balloon flower",
    "giant white arum lily","fire lily","pincushion flower","fritillary","red ginger",
    "grape hyacinth","corn poppy","prince of wales feathers","stemless gentian",
    "artichoke","sweet william","carnation","garden phlox","love in the mist",
    "mexican aster","alpine sea holly","ruby-lipped cattleya","cape flower",
    "great masterwort","siam tulip","lenten rose","barbeton daisy","daffodil",
    "sword lily","poinsettia","bolero deep blue","wallflower","marigold",
    "buttercup","oxeye daisy","common dandelion","petunia","wild pansy",
    "primula","sunflower","pelargonium","bishop of llandaff","gaura","geranium",
    "orange dahlia","pink-yellow dahlia","cautleya spicata","japanese anemone",
    "black-eyed susan","silverbush","californian poppy","osteospermum",
    "spring crocus","bearded iris","windflower","tree poppy","gazania",
    "azalea","water lily","rose","thorn apple","morning glory","passion flower",
    "lotus","toad lily","anthurium","frangipani","clematis","hibiscus",
    "columbine","desert-rose","tree mallow","magnolia","cyclamen","watercress",
    "canna lily","hippeastrum","bee balm","ball moss","foxglove","bougainvillea",
    "camellia","mallow","mexican petunia","bromelia","blanket flower",
    "trumpet creeper","blackberry lily",
]

PROMPT_TEMPLATES = [
    "a photo of {name} flower",
    "a beautiful {name}",
    "a close-up of {name} flower",
    "a {name} in bloom",
    "{name} flower photography",
]

# ─────────────────────────────────────────────────────────────
# 2. 数据工程（8:1:1划分 + 类别分布可视化）
# ─────────────────────────────────────────────────────────────
def build_transforms(split="train", image_size=128):
    if split == "train":
        return T.Compose([
            T.Resize((image_size + 16, image_size + 16),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.08),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

class NamedFlowers(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.ds = base_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        img, label = self.ds[i]
        name = FLOWER_CLASSES[label] if label < len(FLOWER_CLASSES) else f"flower_{label}"
        return img, label, name

def build_dataloader(cfg, split="train"):
    print(f"\n[数据] 下载/加载 Oxford 102 Flowers ({split}) ...")
    subsets_raw = []
    total = 0
    for sp in ["train", "val", "test"]:
        try:
            ds = torchvision.datasets.Flowers102(
                root=cfg["data_dir"], split=sp,
                transform=build_transforms("train" if split == "train" else "val",
                                           cfg["image_size"]),
                download=True
            )
            subsets_raw.append(ds)
            total += len(ds)
            print(f"  {sp:5s}: {len(ds)} 张")
        except Exception as e:
            print(f"  {sp} 加载失败: {e}")

    if not subsets_raw:
        raise RuntimeError("Flowers102 下载失败，请检查网络或 torchvision 版本")

    combined = ConcatDataset(subsets_raw)
    print(f"  合计: {total} 张")

    # 8:1:1 手动划分（固定种子保证可复现）
    gen = torch.Generator().manual_seed(SEED)
    n_train = int(total * 0.8)
    n_val   = int(total * 0.1)
    n_test  = total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(combined, [n_train, n_val, n_test], generator=gen)
    print(f"  划分 → Train:{n_train} | Val:{n_val} | Test:{n_test}")

    chosen = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    named  = NamedFlowers(chosen)

    loader = DataLoader(
        named,
        batch_size=cfg["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"  [{split}] batch_size={cfg['batch_size']}, steps/epoch={len(loader)}")
    return loader, (n_train, n_val, n_test)

def visualize_dataset_stats(split_sizes, save_dir):
    n_train, n_val, n_test = split_sizes
    total = n_train + n_val + n_test

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Oxford 102 Flowers — 数据集统计", fontsize=14, fontweight="bold")

    # 饼图
    ax = axes[0]
    sizes  = [n_train, n_val, n_test]
    labels = [f"Train\n{n_train}张\n({n_train/total*100:.1f}%)",
              f"Val\n{n_val}张\n({n_val/total*100:.1f}%)",
              f"Test\n{n_test}张\n({n_test/total*100:.1f}%)"]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    ax.pie(sizes, labels=labels, colors=colors, startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title(f"Train/Val/Test 划分（总计 {total} 张）")

    # 条形图
    ax = axes[1]
    bars = ax.bar(["Train", "Val", "Test"], sizes, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(v), ha="center", fontweight="bold")
    ax.set_title("各划分样本数量")
    ax.set_ylabel("样本数")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(sizes) * 1.15)

    path = os.path.join(save_dir, "dataset_statistics.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[数据] 统计图已保存 → {path}")
    return path

# ─────────────────────────────────────────────────────────────
# 3. BERT 条件编码器
# ─────────────────────────────────────────────────────────────
class BertConditionEncoder(nn.Module):
    def __init__(self, condition_dim=128, freeze_layers=8,
                 bert_model="bert-base-uncased"):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(bert_model)
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(256, condition_dim), nn.Tanh(),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state[:, 0, :])

def precompute_class_embeddings(bert_enc, tokenizer, names, device):
    bert_enc.eval()
    result = {}
    with torch.no_grad():
        for name in tqdm(names, desc="预计算类别向量(多模板平均)", leave=False):
            vecs = []
            for tmpl in PROMPT_TEMPLATES:
                text = tmpl.format(name=name)
                enc = tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=32)
                enc = {k: v.to(device) for k, v in enc.items()}
                v = bert_enc(enc["input_ids"], enc["attention_mask"]).squeeze(0)
                vecs.append(v)
            result[name] = torch.stack(vecs).mean(0).cpu()
    bert_enc.train()
    return result

# ─────────────────────────────────────────────────────────────
# 4. GAN 模型（高质量生成器 + 预训练判别器）
# ─────────────────────────────────────────────────────────────
def _w(m):
    cn = m.__class__.__name__  # 这一行是关键，定义了变量cn
    if "Conv" in cn:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif "BatchNorm" in cn:
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.constant_(m.bias, val=0)

def _make_norm(norm_type, num_features):
    if norm_type == "batch":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    else:
        raise ValueError(f"未知 norm_type: {norm_type}")

class CondGenerator(nn.Module):
    """高质量DCGAN生成器，保证128x128图片清晰度"""
    def __init__(self, nz=128, cdim=128, ngf=128, nc=3, isize=128, norm_type="batch"):
        super().__init__()
        in_dim = nz + cdim
        layers = [
            nn.ConvTranspose2d(in_dim, ngf*8, 4, 1, 0, bias=False),
            _make_norm(norm_type, ngf*8),
            nn.ReLU(True)
        ]
        ch, s = ngf*8, 4
        while s < isize // 2:
            nxt = max(ch // 2, ngf)
            layers += [
                nn.ConvTranspose2d(ch, nxt, 4, 2, 1, bias=False),
                _make_norm(norm_type, nxt),
                nn.ReLU(True)
            ]
            ch, s = nxt, s * 2
        layers += [
            nn.ConvTranspose2d(ch, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*layers)
        self.apply(_w)

    def forward(self, z, c):
        x = torch.cat([z, c], 1).unsqueeze(-1).unsqueeze(-1)
        return self.net(x)

class PretrainedDiscriminator(nn.Module):
    """预训练ResNet18判别器骨干"""
    def __init__(self, cdim=128, dropout_p=0.3):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = 512
        self.cond_proj = nn.Linear(cdim, feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 1),
        )

    def forward(self, img, cond):
        feat = self.backbone(img).flatten(1)
        c    = self.cond_proj(cond)
        return self.head(torch.cat([feat, c], 1)).squeeze(1)

# ─────────────────────────────────────────────────────────────
# 5. 硬件效率监控工具
# ─────────────────────────────────────────────────────────────
class PerfProfiler:
    def __init__(self, dev):
        self.dev = dev
        self.records = []
    def reset_peak(self):
        if self.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.dev)
    def snapshot(self, label, epoch_time_s, amp_enabled):
        if self.dev.type == "cuda":
            peak_mb = torch.cuda.max_memory_reserved(self.dev) / 1024**2
            cur_mb  = torch.cuda.memory_reserved(self.dev) / 1024**2
        else:
            peak_mb = cur_mb = 0.0
        rec = {
            "label":     label,
            "epoch_s":   round(epoch_time_s, 2),
            "peak_mb":   round(peak_mb, 1),
            "cur_mb":    round(cur_mb, 1),
            "amp":       "FP16(AMP)" if amp_enabled else "FP32",
        }
        self.records.append(rec)
        return rec
    def print_last(self):
        r = self.records[-1]
        print(f"  [Profile] {r['label']} | 耗时={r['epoch_s']:.1f}s | "
              f"显存峰值={r['peak_mb']:.0f}MB | 当前={r['cur_mb']:.0f}MB | 精度={r['amp']}")
    def save_report(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("label,epoch_s,peak_mb,cur_mb,amp\n")
            for r in self.records:
                f.write(f"{r['label']},{r['epoch_s']},{r['peak_mb']},"
                        f"{r['cur_mb']},{r['amp']}\n")
        print(f"[Profile] 报告已保存 → {path}")

# ─────────────────────────────────────────────────────────────
# 6. 训练循环（单组超参数）
# ─────────────────────────────────────────────────────────────
def train_gan_run(hparams, bert_enc, tokenizer, class_embs, split_sizes, profiler, global_step, writer):
    cfg = {**BASE_CFG, **hparams}
    run_id = cfg["run_id"]
    amp = cfg["amp"]

    print(f"\n{'─'*55}")
    print(f"  超参数组 [{run_id}]: {cfg['description']}")
    print(f"  optimizer={cfg['optimizer']} | norm={cfg['norm']} | "
          f"lr={cfg['gan_lr']:.1e} | dropout={cfg['dropout']}")
    print(f"  epochs={cfg['epochs']} | batch_size={cfg['batch_size']} | AMP={amp}")
    print(f"{'─'*55}")

    loader, _ = build_dataloader(cfg, split="train")

    G = CondGenerator(cfg["nz"], cfg["condition_dim"], cfg["ngf"],
                      cfg["channels"], cfg["image_size"],
                      norm_type=cfg["norm"]).to(device)
    D = PretrainedDiscriminator(cfg["condition_dim"],
                                dropout_p=cfg["dropout"]).to(device)

    d_lr = cfg["gan_lr"] * cfg.get("d_lr_scale", 1.0)
    if cfg["optimizer"] == "Adam":
        opt_G = torch.optim.Adam(
            list(G.parameters()) + [p for p in bert_enc.parameters() if p.requires_grad],
            lr=cfg["gan_lr"], betas=(0.5, 0.999)
        )
        opt_D = torch.optim.Adam([
            {"params": D.backbone.parameters(),  "lr": d_lr * 0.1},
            {"params": list(D.cond_proj.parameters()) + list(D.head.parameters()),
             "lr": d_lr},
        ], betas=(0.5, 0.999))
    else:
        opt_G = torch.optim.RMSprop(
            list(G.parameters()) + [p for p in bert_enc.parameters() if p.requires_grad],
            lr=cfg["gan_lr"]
        )
        opt_D = torch.optim.RMSprop([
            {"params": D.backbone.parameters(),  "lr": d_lr * 0.1},
            {"params": list(D.cond_proj.parameters()) + list(D.head.parameters()),
             "lr": d_lr},
        ])

    scaler_G = GradScaler(enabled=amp)
    scaler_D = GradScaler(enabled=amp)
    cond_cache_dev = {n: e.to(device) for n, e in class_embs.items()}

    resume_path = os.path.join(DIRS["ckpt"], f"{run_id}_resume.pt")
    start_ep = 1
    step = global_step
    if os.path.exists(resume_path):
        print(f"  [续训] 检测到断点，从上次中断处继续...")
        ckpt = torch.load(resume_path, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        bert_enc.load_state_dict(ckpt["bert"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_ep = ckpt["epoch"] + 1
        step = ckpt.get("step", step)
        print(f"  [续训] 从 Epoch {start_ep} 继续")

    exp_log = {"run_id": run_id, "hparams": hparams, "epoch_records": []}

    for ep in range(start_ep, cfg["epochs"] + 1):
        profiler.reset_peak()
        t0 = time.time()

        G.train(); D.train(); bert_enc.train()
        dloss_sum = gloss_sum = 0.0
        pbar = tqdm(loader, desc=f"{run_id} Epoch {ep}/{cfg['epochs']}")

        for imgs, _, names in pbar:
            imgs = imgs.to(device); bs = imgs.size(0)
            cond = torch.stack([cond_cache_dev.get(n, torch.zeros(
                cfg["condition_dim"], device=device)) for n in names]).to(device)
            rl = torch.ones(bs,  device=device)
            fl = torch.zeros(bs, device=device)

            # 训练判别器
            opt_D.zero_grad()
            with autocast(enabled=amp):
                z    = torch.randn(bs, cfg["nz"], device=device)
                fake = G(z, cond).detach()
                ld   = F.binary_cross_entropy_with_logits(D(imgs, cond), rl) + \
                       F.binary_cross_entropy_with_logits(D(fake, cond), fl)
            scaler_D.scale(ld).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # 训练生成器
            opt_G.zero_grad()
            with autocast(enabled=amp):
                z    = torch.randn(bs, cfg["nz"], device=device)
                fake = G(z, cond)
                lg   = F.binary_cross_entropy_with_logits(D(fake, cond), rl)
            scaler_G.scale(lg).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            dloss_sum += ld.item(); gloss_sum += lg.item(); step += 1
            pbar.set_postfix(D=f"{ld.item():.3f}", G=f"{lg.item():.3f}")
            writer.add_scalar(f"{run_id}/D_loss", ld.item(), step)
            writer.add_scalar(f"{run_id}/G_loss", lg.item(), step)

        epoch_time = time.time() - t0
        rec = profiler.snapshot(f"{run_id}_ep{ep:02d}", epoch_time, amp)
        profiler.print_last()

        avg_d = dloss_sum / len(loader)
        avg_g = gloss_sum / len(loader)
        print(f"  [{run_id}] Epoch {ep:2d}/{cfg['epochs']} | D={avg_d:.4f} | G={avg_g:.4f} | t={epoch_time:.1f}s")

        writer.add_scalar(f"{run_id}/D_loss_epoch", avg_d, ep)
        writer.add_scalar(f"{run_id}/G_loss_epoch", avg_g, ep)
        if device.type == "cuda":
            writer.add_scalar(f"{run_id}/peak_mem_MB", rec["peak_mb"], ep)

        exp_log["epoch_records"].append({
            "epoch": ep, "D_loss": round(avg_d, 4),
            "G_loss": round(avg_g, 4), "time_s": round(epoch_time, 2), "peak_mb": rec["peak_mb"],
        })

        if ep % cfg["sample_every"] == 0 or ep == cfg["epochs"]:
            G.eval()
            with torch.no_grad():
                sc  = torch.stack(list(cond_cache_dev.values())[:8]).to(device)
                sn  = torch.randn(8, cfg["nz"], device=device)
                out = G(sn, sc)
            save_image(out*.5+.5,
                       os.path.join(DIRS["samples"], f"{run_id}_ep{ep:03d}.png"), nrow=4)
            writer.add_image(f"{run_id}/generated", make_grid(out*.5+.5, 4), ep)
            G.train()

        torch.save({"G": G.state_dict(), "D": D.state_dict(),
                    "bert": bert_enc.state_dict(),
                    "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                    "epoch": ep, "step": step, "cfg": cfg},
                   os.path.join(DIRS["ckpt"], f"{run_id}_resume.pt"))

        if ep % cfg["save_every"] == 0 or ep == cfg["epochs"]:
            torch.save({"G_state": G.state_dict(), "D_state": D.state_dict(),
                        "bert_state": bert_enc.state_dict(), "cfg": cfg},
                       os.path.join(DIRS["ckpt"], f"{run_id}_ep{ep:03d}.pt"))
            print(f"  [保存] {run_id}_ep{ep:03d}.pt")

    torch.save({"G_state": G.state_dict(), "D_state": D.state_dict(),
                "bert_state": bert_enc.state_dict(), "cfg": cfg},
               os.path.join(DIRS["ckpt"], f"{run_id}_final.pt"))
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"  [完成] {run_id} → {DIRS['ckpt']}/{run_id}_final.pt")

    log_path = os.path.join(DIRS["reports"], f"{run_id}_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(exp_log, f, ensure_ascii=False, indent=2)
    print(f"  [记录] 实验日志 → {log_path}")

    return step, exp_log

# ─────────────────────────────────────────────────────────────
# 7. 超参数对比可视化
# ─────────────────────────────────────────────────────────────
def plot_comparison(all_logs, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("三组超参数对比实验 — 训练曲线", fontsize=13, fontweight="bold")
    colors = ["#E53935", "#1E88E5", "#43A047"]
    metrics = [("D_loss", "D Loss"), ("G_loss", "G Loss")]

    for ax, (key, title) in zip(axes, metrics):
        for i, (log, color) in enumerate(zip(all_logs, colors)):
            epochs = [r["epoch"] for r in log["epoch_records"]]
            vals   = [r[key]     for r in log["epoch_records"]]
            run_id = log["run_id"]
            desc   = HPARAM_GROUPS[i]["description"]
            ax.plot(epochs, vals, color=color, linewidth=2,
                    marker="o", markersize=4, label=f"{run_id}: {desc}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "hyperparams_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[汇总] 超参数对比图 → {path}")

def print_comparison_table(all_logs):
    print(f"\n{'='*80}")
    print(f"  超参数对比实验汇总表")
    print(f"{'='*80}")
    print(f"{'Run':<10} {'描述':<35} {'最终D_loss':<12} {'最终G_loss':<12}")
    print(f"{'─'*80}")
    for log in all_logs:
        records = log["epoch_records"]
        last = records[-1]
        hp = next(h for h in HPARAM_GROUPS if h["run_id"] == log["run_id"])
        print(f"{log['run_id']:<10} {hp['description']:<35} "
              f"{last['D_loss']:<12.4f} {last['G_loss']:<12.4f}")
    print(f"{'='*80}")

    csv_path = os.path.join(DIRS["reports"], "comparison_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("run_id,description,optimizer,norm,lr,dropout,final_D_loss,final_G_loss,total_epochs\n")
        for log in all_logs:
            records = log["epoch_records"]
            last    = records[-1]
            hp      = next(h for h in HPARAM_GROUPS if h["run_id"] == log["run_id"])
            f.write(f"{log['run_id']},{hp['description']},"
                    f"{hp['optimizer']},{hp['norm']},{hp['gan_lr']},"
                    f"{hp['dropout']},{last['D_loss']:.4f},{last['G_loss']:.4f},{hp['epochs']}\n")
    print(f"[汇总] 对比表格 → {csv_path}")

# ─────────────────────────────────────────────────────────────
# 8. 主函数
# ─────────────────────────────────────────────────────────────
def main():
    print(f"\n[总览] 三组超参数对比，各训练 10 epoch，共 30 epoch")
    print(f"[总览] 输出目录: {_BASE}")

    # 数据工程
    _loader, split_sizes = build_dataloader(
        {**BASE_CFG, "batch_size": HPARAM_GROUPS[0]["batch_size"]}, split="train"
    )
    visualize_dataset_stats(split_sizes, DIRS["reports"])

    # BERT
    from transformers import BertTokenizer
    print(f"\n[BERT] 加载预训练模型: {BASE_CFG['bert_model']}")
    tokenizer = BertTokenizer.from_pretrained(BASE_CFG["bert_model"])
    bert_enc  = BertConditionEncoder(
        BASE_CFG["condition_dim"], BASE_CFG["freeze_bert_layers"], BASE_CFG["bert_model"]
    ).to(device)

    # 预计算类别向量
    print("[预计算] 102个类别的 BERT 条件向量 ...")
    class_embs = precompute_class_embeddings(bert_enc, tokenizer, FLOWER_CLASSES[:102], device)
    torch.save(class_embs, os.path.join(DIRS["ckpt"], "class_embeddings.pt"))
    class_embs_cpu = {n: e.cpu() for n, e in class_embs.items()}

    # TensorBoard
    writer = SummaryWriter(DIRS["tb"])
    profiler = PerfProfiler(device)

    # 三组超参数训练
    global_step = 0
    all_logs = []
    for hparams in HPARAM_GROUPS:
        run_id = hparams["run_id"]
        final_path = os.path.join(DIRS["ckpt"], f"{run_id}_final.pt")
        if os.path.exists(final_path):
            print(f"\n[跳过] 检测到 {run_id}_final.pt，已训练完成")
            log_path = os.path.join(DIRS["reports"], f"{run_id}_log.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    all_logs.append(json.load(f))
            continue

        global_step, exp_log = train_gan_run(
            hparams, bert_enc, tokenizer, class_embs_cpu,
            split_sizes, profiler, global_step, writer
        )
        all_logs.append(exp_log)
        torch.cuda.empty_cache()

    # 汇总报告
    if len(all_logs) >= 2:
        plot_comparison(all_logs, DIRS["reports"])
        print_comparison_table(all_logs)

    profiler.save_report(os.path.join(DIRS["reports"], "profile_table.csv"))

    writer.close()
    print(f"\n{'='*55}")
    print("  全部训练完成！")
    print(f"  检查点: {DIRS['ckpt']}/")
    print(f"  样本图: {DIRS['samples']}/")
    print(f"  报告:    {DIRS['reports']}/")
    print(f"  TensorBoard: tensorboard --logdir {DIRS['tb']}")
    print(f"  Gradio 展示: python app.py")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()