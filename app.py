"""
app.py — 本地展示端：文本输入 → BERT → cGAN / DDPM 双路生成 → CLIP/BLIP 评估
运行: python app.py
访问: http://127.0.0.1:7860
依赖: pip install torch torchvision transformers diffusers gradio
      pip install open_clip_torch Pillow tqdm
      pip install git+https://github.com/salesforce/BLIP  (可选, 没有自动跳过)
"""

import os, sys, json, time, warnings, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import gradio as gr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. 路径与设备
# ─────────────────────────────────────────────────────────────
# 与 train.py 保持一致的目录结构
_BASE      = os.path.dirname(os.path.abspath(__file__))
GAN_DIR    = os.path.join(_BASE, "checkpoints", "gan")
SHARED_DIR = os.path.join(_BASE, "checkpoints", "shared")

GAN_CKPT   = os.path.join(GAN_DIR,    "gan_ep010.pt")
CLASS_EMB  = os.path.join(SHARED_DIR, "class_embeddings.pt")
CLASS_JSON = os.path.join(SHARED_DIR, "class_names.json")
CFG_JSON   = os.path.join(SHARED_DIR, "train_cfg.json")

# SD v1.5 完整模型目录（下载整个 stable-diffusion-v1-5 文件夹放这里）
SD_MODEL_DIR = os.path.join(_BASE, "sd_model")   # 整个 SD v1.5 目录
# 兼容旧方式：单独 unet + vae
SD_UNET_DIR  = os.path.join(_BASE, "model")
SD_VAE_DIR   = os.path.join(_BASE, "vae_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[设备] {device}")

# ─────────────────────────────────────────────────────────────
# 1. 重新定义模型（与 train.py 完全一致，避免 import 依赖）
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 1. SD v1.5 推理器（替代自训练DDPM）
#    UNet: modelscope stable-diffusion-v1-5/unet
#    VAE:  stabilityai/sd-vae-ft-mse
# ─────────────────────────────────────────────────────────────
class SDInference:
    """
    SD v1.5 推理器。
    优先使用完整 sd_model 目录（StableDiffusionPipeline），
    回退到单独 unet+vae 目录。
    """
    def __init__(self, model_dir, unet_dir, vae_dir, device="cpu"):
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CLIPTokenizer, CLIPTextModel

        self.device = device
        self.vae_scale = 0.18215

        if os.path.exists(os.path.join(model_dir, "model_index.json")):
            # ── 方案A：完整 Pipeline 目录，最简单 ──
            print(f"  加载完整 SD v1.5 Pipeline: {model_dir}")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_dir,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            self.unet         = pipe.unet.eval()
            self.vae          = pipe.vae.eval()
            self.tokenizer    = pipe.tokenizer
            self.text_encoder = pipe.text_encoder.eval()
            self.scheduler    = DDIMScheduler.from_config(pipe.scheduler.config)
            del pipe
            print(f"  ✓ Pipeline 加载成功")

        else:
            # ── 方案B：单独 unet + vae + tokenizer ──
            print(f"  加载 SD UNet: {unet_dir}")
            self.unet = UNet2DConditionModel.from_pretrained(
                unet_dir, local_files_only=True
            ).to(device).eval()

            print(f"  加载 SD VAE: {vae_dir}")
            self.vae = AutoencoderKL.from_pretrained(
                vae_dir, local_files_only=True
            ).to(device).eval()

            # text_encoder 从 sd_model/text_encoder 或在线获取
            text_enc_dir = os.path.join(model_dir, "text_encoder")
            tok_dir      = os.path.join(model_dir, "tokenizer")
            if os.path.exists(text_enc_dir):
                print(f"  加载 text_encoder: {text_enc_dir}")
                self.tokenizer    = CLIPTokenizer.from_pretrained(tok_dir, local_files_only=True)
                self.text_encoder = CLIPTextModel.from_pretrained(text_enc_dir, local_files_only=True).to(device).eval()
            else:
                print(f"  ⚠ 未找到 text_encoder，尝试在线加载...")
                self.tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
                self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
            )

    def encode_text(self, text: str, n_samples: int):
        tokens = self.tokenizer(
            [text]*n_samples, padding="max_length",
            max_length=77, truncation=True, return_tensors="pt"
        ).to(self.device)
        unc_tokens = self.tokenizer(
            [""]*n_samples, padding="max_length",
            max_length=77, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            ctx   = self.text_encoder(**tokens).last_hidden_state
            uncond = self.text_encoder(**unc_tokens).last_hidden_state
        return ctx, uncond

    @torch.no_grad()
    def generate(self, text, n_samples=2, steps=50,
                 guidance_scale=7.5, image_size=512):
        """image_size 建议用 512（SD v1.5 原生分辨率）"""
        ctx, uncond = self.encode_text(text, n_samples)
        lat_h = lat_w = image_size // 8
        latents = torch.randn(n_samples, 4, lat_h, lat_w,
                              device=self.device,
                              dtype=self.unet.dtype)
        self.scheduler.set_timesteps(steps)
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.scheduler.timesteps:
            lat_in = torch.cat([latents]*2)
            enc    = torch.cat([uncond, ctx])
            t_in   = t.expand(n_samples*2).to(self.device)
            noise_pred = self.unet(lat_in, t_in,
                                   encoder_hidden_states=enc).sample
            up, cp = noise_pred.chunk(2)
            noise_pred = up + guidance_scale*(cp - up)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.vae.decode(latents.float() / self.vae_scale).sample.clamp(-1,1)
        results = []
        for i in range(n_samples):
            arr = ((imgs[i].cpu().float().numpy().transpose(1,2,0)+1)/2*255
                   ).clip(0,255).astype("uint8")
            results.append(Image.fromarray(arr))
        return results


sd_inference = None   # 延迟初始化

def _load_sd():
    global sd_inference
    if sd_inference is not None:
        return True
    # 优先检查完整 SD v1.5 模型目录，其次检查旧方式（单独 unet + vae）
    has_full_model = os.path.exists(os.path.join(SD_MODEL_DIR, "model_index.json"))
    has_old_model = os.path.exists(SD_UNET_DIR) and os.path.exists(SD_VAE_DIR)
    
    if not has_full_model and not has_old_model:
        print(f"  ⚠ SD模型目录不存在:")
        print(f"    - 完整模型: {SD_MODEL_DIR} (需包含 model_index.json)")
        print(f"    - 旧方式: {SD_UNET_DIR} / {SD_VAE_DIR}")
        return False
    try:
        sd_inference = SDInference(
            model_dir=SD_MODEL_DIR,
            unet_dir=SD_UNET_DIR,
            vae_dir=SD_VAE_DIR,
            device=device,
        )
        print("  ✓ SD v1.5 推理器加载成功")
        return True
    except Exception as e:
        print(f"  ⚠ SD加载失败: {e}")
        return False

class ConditionalGenerator(nn.Module):
    """cGAN 生成器，与 train.py 完全一致"""
    def __init__(self, nz=100, condition_dim=128, ngf=64, nc=3, image_size=64):
        super().__init__()
        in_dim = nz + condition_dim
        layers = [nn.ConvTranspose2d(in_dim, ngf*8, 4,1,0,bias=False),
                  nn.BatchNorm2d(ngf*8), nn.ReLU(True)]
        ch, s = ngf*8, 4
        while s < image_size // 2:
            nxt = max(ch//2, ngf)
            layers += [nn.ConvTranspose2d(ch,nxt,4,2,1,bias=False),
                       nn.BatchNorm2d(nxt), nn.ReLU(True)]
            ch, s = nxt, s*2
        layers += [nn.ConvTranspose2d(ch, nc, 4,2,1,bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self, z, c):
        x = torch.cat([z, c], 1).unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


class BertConditionEncoder(nn.Module):
    def __init__(self, condition_dim=128, freeze_layers=8, bert_model="bert-base-uncased"):
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
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)


class DDPMScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.betas = betas; self.alphas = alphas; self.alphas_cumprod = ac

    @torch.no_grad()
    def sample(self, model, condition, shape, device, steps=200):
        """加速采样：只跑 steps 步（DDIM 风格跳步）"""
        model.eval()
        x = torch.randn(shape, device=device)
        skip = self.T // steps
        seq = list(range(0, self.T, skip))[::-1]
        for t in seq:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = model(x, t_batch, condition)
            beta = self.betas[t]
            alpha = self.alphas[t]
            ac = self.alphas_cumprod[t]
            x = (x - (beta / (1 - ac).sqrt()) * eps) / alpha.sqrt()
            if t > 0:
                x += beta.sqrt() * 0.5 * torch.randn_like(x)
        model.train()
        return x.clamp(-1, 1)


# ─────────────────────────────────────────────────────────────
# 2. 加载已训练的模型
# ─────────────────────────────────────────────────────────────
def load_cfg():
    if os.path.exists(CFG_JSON):
        with open(CFG_JSON) as f:
            return json.load(f)
    return {
        "nz": 100, "condition_dim": 128, "ngf": 64, "ndf": 64,
        "channels": 3, "image_size": 64, "ddpm_channels": 64,
        "timesteps": 1000, "bert_model": "bert-base-uncased",
        "freeze_bert_layers": 8,
    }


CFG = load_cfg()


def load_class_names():
    if os.path.exists(CLASS_JSON):
        with open(CLASS_JSON) as f:
            return json.load(f)
    return [f"class_{i}" for i in range(102)]


CLASS_NAMES = load_class_names()


# 强制离线模式：直接用本地缓存，不尝试连接 HuggingFace
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

print("\n[加载] 初始化 BERT tokenizer ...")
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(CFG["bert_model"])

print("[加载] BERT 条件编码器 ...")
bert_encoder = BertConditionEncoder(
    condition_dim=CFG["condition_dim"],
    freeze_layers=CFG.get("freeze_bert_layers", 8),
    bert_model=CFG["bert_model"]
).to(device).eval()

print("[加载] cGAN 生成器 ...")
# 先从 checkpoint 读取训练时的真实参数，再重建生成器
if os.path.exists(GAN_CKPT):
    _ckpt = torch.load(GAN_CKPT, map_location="cpu", weights_only=False)
    _saved = _ckpt.get("cfg", {})
    for _k in ["nz", "ngf", "condition_dim", "image_size", "channels"]:
        if _k in _saved:
            CFG[_k] = _saved[_k]
    print(f"  [训练参数] nz={CFG['nz']} ngf={CFG['ngf']} image_size={CFG['image_size']}")

G_net = ConditionalGenerator(
    nz=CFG["nz"], condition_dim=CFG["condition_dim"],
    ngf=CFG.get("ngf", 128), nc=CFG["channels"], image_size=CFG["image_size"]
).to(device).eval()

if os.path.exists(GAN_CKPT):
    G_net.load_state_dict(_ckpt["G_state"])
    bert_encoder.load_state_dict(_ckpt["bert_state"])
    print(f"  ✓ GAN 权重加载成功: {GAN_CKPT}")
else:
    print(f"  ⚠ 未找到 {GAN_CKPT}，请先运行 train.py")

print("[加载] DDPM U-Net (HuggingFace UNet2DModel)...")
print("[加载] SD v1.5 推理器（延迟加载，首次生成时初始化）")
print(f"  UNet 路径: {SD_UNET_DIR}")
print(f"  VAE  路径: {SD_VAE_DIR}")


# 加载类别缓存向量（加速推理）
class_emb_cache = {}
if os.path.exists(CLASS_EMB):
    print(f"[加载] 类别条件向量: {CLASS_EMB}")
    class_emb_cache = torch.load(CLASS_EMB, map_location=device)
    print(f"  ✓ {len(class_emb_cache)} 个类别")


# ─────────────────────────────────────────────────────────────
# 3 & 4. CLIP / BLIP 延迟加载（第一次点击生成时才下载，界面先启动）
# ─────────────────────────────────────────────────────────────
clip_model = None
clip_preprocess = None
clip_tokenizer = None
blip_model = None
blip_processor = None
_clip_loaded = False
_blip_loaded = False

def _load_clip():
    global clip_model, clip_preprocess, clip_tokenizer, _clip_loaded
    if _clip_loaded:
        return
    print("\n[加载] CLIP (ViT-B-32) 从本地缓存加载...")

    import pathlib, open_clip
    clip_local = pathlib.Path.home() / ".cache" / "clip" / "ViT-B-32.pt"

    try:
        # PyTorch 2.6+ weights_only 默认 True，TorchScript .pt 需要 False
        # 用 monkey-patch 临时允许
        import torch as _torch
        _orig_load = _torch.load
        _torch.load = lambda *a, **kw: _orig_load(
            *a, **{**kw, "weights_only": False}
        )
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(clip_local)
        )
        _torch.load = _orig_load  # 恢复原始函数
        clip_model = clip_model.to(device).eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        print(f"  ✓ CLIP 加载成功（本地文件）")
    except Exception as e:
        print(f"  ⚠ CLIP 加载失败: {e}")
    _clip_loaded = True

def _load_blip():
    global blip_model, blip_processor, _blip_loaded
    if _blip_loaded:
        return
    print("\n[加载] BLIP 从本地缓存加载...")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        # local_files_only=True：只读本地缓存，不联网
        blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", local_files_only=True
        )
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", local_files_only=True
        ).to(device).eval()
        print("  ✓ BLIP 加载成功")
    except Exception as e:
        print(f"  ⚠ BLIP 加载失败: {e}")
    _blip_loaded = True

print("\n[跳过] CLIP / BLIP 延迟加载，第一次生成时自动下载")


# ─────────────────────────────────────────────────────────────
# 5. 核心推理函数
# ─────────────────────────────────────────────────────────────
def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[-1,1] tensor [C,H,W] → PIL Image"""
    t = t.squeeze(0).clamp(-1, 1)
    arr = ((t.cpu().float().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
    return Image.fromarray(arr)


# 中文关键词 → 英文 映射表（支持常见中文输入）
ZH_TO_EN = {
    "玫瑰": "rose", "红玫瑰": "rose", "黄玫瑰": "yellow rose",
    "向日葵": "sunflower", "太阳花": "sunflower",
    "黄花": "yellow flower", "黄色的花": "yellow flower",
    "红花": "red flower", "红色的花": "red flower",
    "紫花": "purple flower", "紫色的花": "purple flower",
    "粉花": "pink flower", "粉色的花": "pink flower",
    "白花": "white flower", "蓝花": "blue flower",
    "郁金香": "tulip", "百合": "lily", "雏菊": "daisy",
    "牡丹": "peony", "荷花": "lotus", "薰衣草": "lavender",
    "樱花": "cherry blossom", "水仙": "daffodil",
    "兰花": "orchid", "牵牛花": "morning glory",
    "大丽花": "dahlia", "菊花": "chrysanthemum",
}


def translate_if_chinese(text: str) -> str:
    """简单中英文映射，找到就替换，找不到就原样（BERT能处理英文）"""
    t = text.strip()
    # 直接匹配
    if t in ZH_TO_EN:
        return ZH_TO_EN[t]
    # 部分匹配（逐词检查）
    for zh, en in ZH_TO_EN.items():
        if zh in t:
            t = t.replace(zh, en)
    return t


def get_condition_vector(text: str) -> torch.Tensor:
    """
    用 BERT 将自由文本编码为条件向量。
    支持中文输入（自动映射到英文后送入 BERT）。
    与训练时一致：用多个模板编码后取平均，提升对齐精度。
    """
    text_en = translate_if_chinese(text)

    # 多模板平均（与训练时 precompute_class_embeddings 保持一致）
    templates = [
        f"a photo of {text_en} flower",
        f"a beautiful {text_en}",
        f"a close-up of {text_en} flower",
        f"a {text_en} in bloom",
        f"{text_en} flower photography",
    ]
    vecs = []
    with torch.no_grad():
        for tmpl in templates:
            enc = tokenizer(tmpl, return_tensors="pt",
                            padding=True, truncation=True, max_length=32)
            enc = {k: v.to(device) for k, v in enc.items()}
            v = bert_encoder(enc["input_ids"], enc["attention_mask"])
            vecs.append(v)
    cond = torch.stack(vecs).mean(0)  # [1, condition_dim]
    return cond


def generate_gan(condition: torch.Tensor, n_samples: int = 4) -> list:
    """cGAN 生成，nz 从 CFG 读取，与训练时保持一致"""
    nz = CFG.get("nz", 128)
    G_net.eval()
    with torch.no_grad():
        cond_rep = condition.repeat(n_samples, 1)
        noise = torch.randn(n_samples, nz, device=device)
        imgs = G_net(noise, cond_rep)  # [N, C, H, W]
    # 调试：输出tensor统计信息
    print(f"  [GAN调试] imgs shape={imgs.shape} min={imgs.min():.3f} max={imgs.max():.3f} mean={imgs.mean():.3f}")
    return [tensor_to_pil(imgs[i]) for i in range(n_samples)]


def generate_ddpm(text: str, n_samples: int = 2,
                  ddpm_steps: int = 50) -> list:
    """SD v1.5 推理，直接用文本字符串通过 CLIP 编码条件"""
    if not _load_sd():
        blank = Image.new("RGB", (CFG["image_size"], CFG["image_size"]), (40, 40, 60))
        return [blank] * n_samples
    return sd_inference.generate(
        text, n_samples=n_samples,
        steps=ddpm_steps,
        guidance_scale=7.5,
        image_size=512,   # SD v1.5 原生分辨率，生成质量最佳
    )


def clip_zero_shot(pil_images: list, candidate_labels: list) -> list[dict]:
    """CLIP 零样本分类，返回每张图的 Top-5 类别及得分"""
    if clip_model is None:
        return [{"error": "CLIP 未加载"} for _ in pil_images]

    results = []
    try:
        import open_clip
        text_tokens = clip_tokenizer(
            [f"a photo of {c}" for c in candidate_labels]
        ).to(device)
    except Exception:
        import clip as _clip
        text_tokens = _clip.tokenize(
            [f"a photo of {c}" for c in candidate_labels]
        ).to(device)

    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        for pil in pil_images:
            img_t = clip_preprocess(pil).unsqueeze(0).to(device)
            img_feat = clip_model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ text_feats.T).squeeze(0)
            probs = sims.softmax(dim=-1).cpu().numpy()
            top5_idx = probs.argsort()[::-1][:5]
            results.append({
                candidate_labels[i]: float(f"{probs[i]*100:.1f}")
                for i in top5_idx
            })
    return results


def clip_text_image_similarity(pil_image: Image.Image, text: str) -> float:
    """计算图像与输入文本的 CLIP 余弦相似度"""
    if clip_model is None:
        return 0.0
    try:
        import open_clip
        text_tokens = clip_tokenizer([text]).to(device)
    except Exception:
        import clip as _clip
        text_tokens = _clip.tokenize([text]).to(device)

    with torch.no_grad():
        img_t = clip_preprocess(pil_image).unsqueeze(0).to(device)
        img_feat = clip_model.encode_image(img_t)
        txt_feat = clip_model.encode_text(text_tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T).item()
    return round(sim, 4)


def blip_caption(pil_image: Image.Image) -> str:
    """BLIP 图像描述生成"""
    if blip_model is None or blip_processor is None:
        return "BLIP 未加载（pip install transformers 后重启）"
    try:
        inputs = blip_processor(pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=50)
        return blip_processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"BLIP 生成失败: {e}"


# ─────────────────────────────────────────────────────────────
# 6. Gradio 主界面回调
# ─────────────────────────────────────────────────────────────
def run_generation(
    text_prompt: str,
    n_gan: int,
    n_ddpm: int,
    ddpm_steps: int,
    top_k_classes: int,
    show_clip: bool,
    show_blip: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not text_prompt.strip():
        return ([], [], "❌ 请输入文本描述", "", "", "")

    logs = []
    t0 = time.time()

    # ── 按需加载 SD / CLIP / BLIP（首次调用时初始化）──
    _load_sd()   # SD 推理器（本地模型，秒加载）
    if show_clip:
        _load_clip()
    if show_blip:
        _load_blip()

    # ── BERT 编码 ──
    logs.append(f"[BERT] 编码文本: '{text_prompt}'")
    progress(0.1, desc="BERT 文本编码中 ...")
    cond = get_condition_vector(text_prompt)
    logs.append(f"  条件向量维度: {cond.shape} | 均值: {cond.mean().item():.4f}")

    # ── GAN 生成 ──
    logs.append(f"\n[cGAN] 生成 {n_gan} 张图像 ...")
    progress(0.25, desc="cGAN 生成中 ...")
    gan_images = generate_gan(cond, n_samples=n_gan)
    logs.append(f"  完成，耗时 {time.time()-t0:.2f}s")

    # ── DDPM 生成 ──
    t1 = time.time()
    logs.append(f"\n[DDPM] 生成 {n_ddpm} 张图像 (采样步数={ddpm_steps}) ...")
    progress(0.4, desc="DDPM 扩散采样中 ...")
    # SD直接用原始文本，不需要BERT条件向量
    ddpm_images = generate_ddpm(text_prompt, n_samples=n_ddpm, ddpm_steps=ddpm_steps)
    logs.append(f"  完成，耗时 {time.time()-t1:.2f}s")

    all_images = gan_images + ddpm_images

    # ── CLIP 零样本分类 ──
    clip_report = ""
    if show_clip and clip_model is not None:
        progress(0.65, desc="CLIP 零样本分类中 ...")
        # 候选标签: 输入文本本身 + 最近的花卉类别
        candidates = CLASS_NAMES[:top_k_classes]
        results = clip_zero_shot(all_images, candidates)

        lines = ["### CLIP 零样本分类结果\n"]
        for i, (img, res) in enumerate(zip(all_images, results)):
            model_tag = "GAN" if i < n_gan else "DDPM"
            idx = i if i < n_gan else i - n_gan
            lines.append(f"**{model_tag} 图像 {idx+1}:**")
            for label, score in res.items():
                bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
                lines.append(f"  `{label:<30}` {bar} {score:.1f}%")
            lines.append("")

        # CLIP 相似度 (vs 输入文本)
        sims_gan  = [clip_text_image_similarity(img, text_prompt) for img in gan_images]
        sims_ddpm = [clip_text_image_similarity(img, text_prompt) for img in ddpm_images]
        lines.append("### CLIP 文本-图像相似度 (vs 输入文本)")
        lines.append(f"- **cGAN**: {[f'{s:.3f}' for s in sims_gan]}")
        lines.append(f"  平均: **{np.mean(sims_gan):.4f}**")
        lines.append(f"- **DDPM**: {[f'{s:.3f}' for s in sims_ddpm]}")
        lines.append(f"  平均: **{np.mean(sims_ddpm):.4f}**")

        winner = "cGAN" if np.mean(sims_gan) >= np.mean(sims_ddpm) else "DDPM"
        lines.append(f"\n🏆 **文本对齐度更优模型: {winner}**")
        clip_report = "\n".join(lines)
        logs.append(f"[CLIP] 分类 + 相似度计算完成")
    elif show_clip:
        clip_report = "⚠ CLIP 未加载，请安装 open_clip_torch"

    # ── BLIP 图像描述 ──
    blip_report = ""
    if show_blip:
        progress(0.85, desc="BLIP 图像描述生成中 ...")
        lines_b = ["### BLIP 图像描述（自动标注）\n"]
        for i, img in enumerate(all_images):
            model_tag = "GAN" if i < n_gan else "DDPM"
            idx = i if i < n_gan else i - n_ddpm
            cap = blip_caption(img)
            lines_b.append(f"**{model_tag} 图像 {i+1}:** `{cap}`")
        blip_report = "\n".join(lines_b)
        logs.append("[BLIP] 图像描述生成完成")

    # ── 摘要 ──
    total_time = time.time() - t0
    summary_lines = [
        f"## 生成摘要",
        f"- **输入文本**: `{text_prompt}`",
        f"- **cGAN 生成**: {n_gan} 张图",
        f"- **DDPM 生成**: {n_ddpm} 张图 ({ddpm_steps} 步采样)",
        f"- **总耗时**: {total_time:.2f}s",
        f"- **设备**: {device}",
    ]
    if device.type == "cuda":
        mem_mb = torch.cuda.memory_allocated() / 1024**2
        summary_lines.append(f"- **显存占用**: {mem_mb:.0f} MB")
    summary = "\n".join(summary_lines)

    progress(1.0, desc="完成！")
    logs.append(f"\n✓ 全部完成，总耗时 {total_time:.2f}s")

    return (
        gan_images,
        ddpm_images,
        "\n".join(logs),
        summary,
        clip_report,
        blip_report,
    )


# ─────────────────────────────────────────────────────────────
# 7. Gradio UI 布局
# ─────────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; font-size: 1.5em; font-weight: bold; margin-bottom: 0.5em; }
#gan_col { border: 2px solid #4a90d9; border-radius: 8px; padding: 10px; }
#ddpm_col { border: 2px solid #e07b39; border-radius: 8px; padding: 10px; }
.label-gan { color: #4a90d9; font-weight: bold; }
.label-ddpm { color: #e07b39; font-weight: bold; }
"""

EXAMPLE_PROMPTS = [
    "a beautiful red rose with dew drops",
    "a vibrant sunflower field in summer",
    "a delicate pink cherry blossom",
    "a purple lavender bunch in morning light",
    "a colorful tulip garden in spring",
    "a white water lily on a calm pond",
    "a blue iris flower with yellow center",
]

with gr.Blocks(
    title="跨模态图像生成系统 — GAN vs DDPM",
    css=CSS,
    theme=gr.themes.Soft(),
) as demo:

    gr.HTML("<div id='title'>🌺 跨模态图像生成系统 — cGAN × DDPM × BERT × CLIP × BLIP</div>")
    gr.Markdown(
        "**输入文本描述** → BERT 提取语义条件 → **cGAN** 和 **DDPM** 同时生成图像 → "
        "CLIP 零样本分类 + 文本对齐评分 → BLIP 自动描述"
    )

    with gr.Row():
        # ── 左侧控制面板 ──
        with gr.Column(scale=1):
            gr.Markdown("### ✏️ 输入设置")
            text_input = gr.Textbox(
                label="文本描述 (英文效果更好)",
                placeholder="例如: a beautiful red rose with dew drops",
                lines=2,
            )
            gr.Examples(
                examples=[[p] for p in EXAMPLE_PROMPTS],
                inputs=[text_input],
                label="示例提示词",
            )
            with gr.Accordion("⚙️ 生成参数", open=True):
                n_gan = gr.Slider(1, 8, value=4, step=1, label="cGAN 生成数量")
                n_ddpm = gr.Slider(1, 4, value=2, step=1, label="DDPM 生成数量")
                ddpm_steps = gr.Slider(
                    20, 500, value=100, step=10,
                    label="DDPM 采样步数 (越多越精细，越慢)"
                )
            with gr.Accordion("🔍 评估设置", open=True):
                top_k = gr.Slider(5, 102, value=20, step=5,
                                  label="CLIP 候选类别数量")
                show_clip = gr.Checkbox(value=True, label="启用 CLIP 评估")
                show_blip = gr.Checkbox(
                    value=(blip_model is not None), label="启用 BLIP 图像描述"
                )
            run_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")

        # ── 右侧结果区域 ──
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ 生成结果对比")
            with gr.Row():
                with gr.Column(elem_id="gan_col"):
                    gr.HTML("<div class='label-gan'>🔵 cGAN 生成</div>")
                    gan_gallery = gr.Gallery(
                        label="cGAN 输出", show_label=False,
                        columns=4, height=480, object_fit="contain"
                    )
                with gr.Column(elem_id="ddpm_col"):
                    gr.HTML("<div class='label-ddpm'>🟠 DDPM 生成</div>")
                    ddpm_gallery = gr.Gallery(
                        label="DDPM 输出", show_label=False,
                        columns=4, height=480, object_fit="contain"
                    )

    with gr.Row():
        summary_md = gr.Markdown(label="生成摘要")

    with gr.Tabs():
        with gr.TabItem("📊 CLIP 评估"):
            clip_md = gr.Markdown("点击生成后查看 CLIP 零样本分类结果")
        with gr.TabItem("💬 BLIP 描述"):
            blip_md = gr.Markdown("点击生成后查看 BLIP 自动描述")
        with gr.TabItem("🔧 运行日志"):
            log_box = gr.Textbox(
                label="系统日志", lines=15,
                max_lines=30, interactive=False
            )

    # ── 绑定事件 ──
    run_btn.click(
        fn=run_generation,
        inputs=[text_input, n_gan, n_ddpm, ddpm_steps, top_k, show_clip, show_blip],
        outputs=[gan_gallery, ddpm_gallery, log_box, summary_md, clip_md, blip_md],
    )

    # Enter 键也触发
    text_input.submit(
        fn=run_generation,
        inputs=[text_input, n_gan, n_ddpm, ddpm_steps, top_k, show_clip, show_blip],
        outputs=[gan_gallery, ddpm_gallery, log_box, summary_md, clip_md, blip_md],
    )

    gr.Markdown(
        "---\n"
        "**系统信息**: "
        f"设备={device} | "
        f"CLIP={'✓' if clip_model else '✗'} | "
        f"BLIP={'✓' if blip_model else '✗'} | "
        f"类别数={len(CLASS_NAMES)}\n\n"
        "> 首次加载模型需要几秒，DDPM 步数越多生成越精细但越慢。"
        "建议先用 50-100 步测试，满意后再提高步数。"
    )


# ─────────────────────────────────────────────────────────────
# 8. 启动
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  跨模态图像生成系统 — 展示端")
    print(f"  设备:  {device}")
    print(f"  CLIP:  {'已加载' if clip_model else '未加载'}")
    print(f"  BLIP:  {'已加载' if blip_model else '未加载'}")
    print(f"  类别数: {len(CLASS_NAMES)}")
    print("="*60)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )