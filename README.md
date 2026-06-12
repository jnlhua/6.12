# 跨模态图像生成系统 🌸

基于条件GAN与Stable Diffusion的花卉图像生成与对比分析系统

## 📋 项目简介

本项目实现了一个完整的跨模态图像生成系统，主要功能包括：

- **条件GAN (cGAN)**：使用BERT文本编码器作为条件，在Oxford Flowers 102数据集上训练高质量花卉图像生成器
- **Stable Diffusion v1.5**：集成预训练扩散模型，提供高质量的文本到图像生成能力
- **对比分析**：通过Gradio Web界面同时展示两种模型的生成效果，并集成CLIP和BLIP进行智能评估
- **超参数实验**：系统性地对比三组不同超参数配置的训练效果

## ✨ 核心特性

### 🎯 模型架构
- **条件生成器 (CondGenerator)**：DCGAN风格的128×128图像生成器
  - 支持BatchNorm和InstanceNorm归一化
  - 噪声向量维度：128维
  - 条件向量维度：128维（来自BERT编码）
  
- **预训练判别器 (PretrainedDiscriminator)**：基于ResNet backbone
  - 可配置dropout率（0.3或0.5）
  - 非对称学习率策略
  
- **BERT条件编码器 (BertConditionEncoder)**：
  - 使用`bert-base-uncased`预训练模型
  - 冻结前8层，微调后层
  - 将花卉名称文本转换为128维条件向量

### 🔬 训练系统
- **三组超参数对比实验**：

| 实验组 | 优化器 | 归一化 | 学习率 | Dropout | Epochs |
|--------|--------|--------|--------|---------|--------|
| Run-A | Adam | BatchNorm | 2e-4 | 0.3 | 10 |
| Run-B | Adam | InstanceNorm | G:1e-4, D:4e-4 | 0.3 | 10 |
| Run-C | RMSprop | BatchNorm | 2e-4 | 0.5 | 10 |

- **训练增强功能**：
  - ✅ AMP混合精度训练（FP16加速）
  - ✅ 梯度裁剪（防止梯度爆炸）
  - ✅ 断点续训（支持中断恢复）
  - ✅ 性能监控（显存、耗时、精度记录）
  - ✅ TensorBoard实时可视化
  - ✅ 自动生成对比报告和图表

### 🖼️ 图像生成与评估
- **双模型并行生成**：
  - cGAN：快速生成（毫秒级），适合批量生成
  - Stable Diffusion：高质量生成（秒级），细节更丰富
  
- **智能评估模块**：
  - **CLIP零样本分类**：自动识别生成图像的花卉类别及置信度
  - **CLIP文本-图像相似度**：量化生成图像与输入文本的语义一致性
  - **BLIP图像描述**：自动生成图像的自然语言描述

### 🌐 Web交互界面
- **Gradio展示平台**：
  - 文本输入框（支持中英文关键词）
  - 参数调节面板（生成数量、采样步数等）
  - 并排对比展示（cGAN vs SD）
  - 实时运行日志
  - CLIP/BLIP评估结果标签页

## 📁 项目结构

```
claude/
├── train.py                 # 主训练脚本（超参数对比实验）
├── app.py                   # Gradio Web应用（图像生成与展示）
├── check_deps.py            # 依赖包版本检查工具
│
├── data/
│   └── flowers102/          # Oxford Flowers 102 数据集
│       └── flowers-102/
│           ├── jpg/         # 图像文件（8000+张）
│           ├── imagelabels.mat
│           └── setid.mat
│
├── sd_model/                # Stable Diffusion v1.5 完整模型
│   ├── model_index.json     # 模型索引配置
│   ├── configuration.json   # 训练配置
│   ├── unet/                # UNet噪声预测网络
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.bin
│   ├── vae/                 # 变分自编码器
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.bin
│   ├── text_encoder/        # CLIP文本编码器
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   ├── tokenizer/           # CLIP分词器
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   └── tokenizer_config.json
│   ├── scheduler/           # PNDM采样调度器
│   │   └── scheduler_config.json
│   └── feature_extractor/   # 特征提取器配置
│
├── checkpoints/
│   ├── gan/                 # cGAN模型检查点
│   │   ├── run_A_final.pt   # Run-A最终模型
│   │   ├── run_B_final.pt   # Run-B最终模型
│   │   ├── run_C_final.pt   # Run-C最终模型
│   │   ├── *_resume.pt      # 断点续训文件
│   │   └── samples/         # 训练过程样本图
│   ├── shared/              # 共享资源
│   │   ├── class_embeddings.pt    # 102类BERT条件向量缓存
│   │   ├── class_names.json       # 102个花卉类别名称
│   │   └── train_cfg.json         # 训练配置记录
│   └── ddpm/                # DDPM模型检查点（可选）
│
├── reports/                 # 实验报告与可视化
│   ├── run_A_log.json       # Run-A训练日志
│   ├── run_B_log.json       # Run-B训练日志
│   ├── run_C_log.json       # Run-C训练日志
│   ├── comparison_table.csv # 三组实验对比表格
│   ├── hyperparams_comparison.png # 超参数对比图表
│   ├── dataset_statistics.png      # 数据集统计图
│   └── profile_table.csv          # 性能监控数据
│
├── runs/                    # TensorBoard日志目录
│   └── events.out.tfevents.*      # 训练事件记录
│
└── .git/                    # Git版本控制
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **CUDA**: 11.0+（推荐用于GPU加速）
- **操作系统**: Windows/Linux/macOS

### 安装依赖

```bash
# 进入项目目录
cd claude

# 安装所有依赖包
pip install torch torchvision transformers diffusers accelerate tensorboardX scipy Pillow tqdm gradio numpy

# 或者使用提供的依赖检查脚本
python check_deps.py
```

### 数据准备

首次运行时，程序会自动下载Oxford Flowers 102数据集：

```bash
# 数据将下载到 data/flowers102/ 目录
# 包含102类花卉，共约8000张图像
python train.py  # 首次运行会自动下载数据
```

### Stable Diffusion模型准备

**方法一：使用完整模型目录（推荐）**
```bash
# 从 HuggingFace 下载 stable-diffusion-v1-5 整个仓库
# 放置到 sd_model/ 目录
# 下载地址：https://huggingface.co/runwayml/stable-diffusion-v1-5
```

**方法二：仅使用UNet + VAE**
```bash
# 下载 UNet 到 model/ 目录
# 下载 VAE 到 vae_model/ 目录
# 程序会自动检测并回退到这种方式
```

## 📖 使用指南

### 1️⃣ 训练模型

```bash
# 运行完整的三组超参数对比实验（共30 epochs）
python train.py

# 训练过程中会：
# - 自动划分数据集（80%训练 / 10%验证 / 10%测试）
# - 预计算102个花卉类别的BERT条件向量
# - 依次训练Run-A/B/C三组实验
# - 每2个epoch保存样本图
# - 每5个epoch保存检查点
# - 生成对比报告和性能监控数据
```

**训练输出位置**：
- 模型权重：`checkpoints/gan/run_*_final.pt`
- 样本图片：`checkpoints/gan/samples/`
- TensorBoard日志：`runs/`
- 对比报告：`reports/`

**查看训练曲线**：
```bash
tensorboard --logdir runs
```

### 2️⃣ 启动Web界面

```bash
# 启动Gradio展示界面
python app.py

# 浏览器访问：http://127.0.0.1:7860
```

**Web界面功能**：
- 输入文本提示词（如"rose"、"向日葵"、"a beautiful sunflower"）
- 选择生成数量（cGAN和SD分别可生成多张）
- 调节DDPM采样步数（50-100步推荐）
- 启用/关闭CLIP和BLIP评估
- 查看实时运行日志
- 对比查看生成结果和评估指标

**支持的中文关键词示例**：
```
玫瑰, 向日葵, 郁金香, 百合, 牡丹, 荷花, 薰衣草, 樱花,
兰花, 大丽花, 菊花, 红花, 黄花, 紫花, 粉花...
```

### 3️⃣ 检查依赖环境

```bash
# 查看已安装的依赖包版本
python check_deps.py

# 输出示例：
# ==================================================
# Claude项目依赖包版本检查
# ==================================================
# Python版本: 3.9.0
# --------------------------------------------------
# torch            → 2.0.0+cu117
# torchvision      → 15.1.0+cu117
# transformers     → 4.30.0
# diffusers        → 0.18.0
# ...
# ✅ 所有依赖已安装!
# ==================================================
```

## 🎨 支持的花卉类别（102类）

系统支持生成102种不同类型的花卉图像，包括但不限于：

**常见花卉**：
- 🌹 Rose (玫瑰)
- 🌻 Sunflower (向日葵)
- 🌷 Tulip (郁金香)
- 🌸 Cherry Blossom (樱花)
- 🪷 Lotus (荷花)
- 🌺 Hibiscus (木槿)
- 🌼 Daisy (雏菊)
- 💐 Lily (百合)

**完整列表**：详见 `checkpoints/shared/class_names.json`

## 📊 实验结果

### 训练性能指标

每组实验会记录以下关键指标：
- **Generator Loss**: 生成器损失值变化
- **Discriminator Loss**: 判别器损失值变化
- **D(G(z))**: 判别器对生成样本的评分
- **Epoch Time**: 每轮训练耗时
- **Peak Memory GPU**: 显存峰值占用
- **Training Precision**: FP16(AMP) 或 FP32

### 可视化报告

训练完成后会自动生成：
- **超参数对比图** (`hyperparams_comparison.png`)
- **数据集统计图** (`dataset_statistics.png`)
- **对比表格** (`comparison_table.csv`)
- **性能监控表** (`profile_table.csv`)

## ⚙️ 高级配置

### 修改超参数

编辑 train.py 中的 `HPARAM_GROUPS` 配置：

```python
HPARAM_GROUPS = [
    {
        "run_id": "run_custom",
        "description": "自定义配置",
        "gan_lr": 2e-4,
        "optimizer": "Adam",  # 可选: Adam, RMSprop
        "norm": "batch",      # 可选: batch, instance
        "dropout": 0.3,       # 推荐: 0.3-0.5
        "epochs": 20,         # 增加训练轮数
        "batch_size": 16,     # 根据显存调整
    },
]
```

### 关键全局配置 (train.py 第46-112行)

```python
BASE_CFG = {
    "image_size":      128,      # 生成图像分辨率
    "channels":        3,        # RGB通道数
    "nz":              128,      # 噪声向量维度
    "condition_dim":   128,      # BERT条件向量维度
    "ngf":             128,      # 生成器基础通道数
    "ndf":             128,      # 判别器基础通道数
    "amp":             True,     # 是否启用混合精度训练
    "bert_model":      "bert-base-uncased",  # BERT预训练模型
    "freeze_bert_layers": 8,     # 冻结BERT层数
}
```

## 🔧 故障排除

### 常见问题

**Q1: CUDA out of memory**
```bash
# 解决方案：减小batch_size
# 编辑 train.py: batch_size = 4  # 或更小
```

**Q2: Stable Diffusion加载失败**
```bash
# 确保sd_model/目录包含完整的模型文件
# 或检查model/和vae_model/目录是否有单独的UNet和VAE
```

**Q3: Flowers102数据集下载失败**
```bash
# 手动下载数据集并解压到 data/flowers102/flowers-102/
# 下载地址：https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
```

**Q4: CLIP/BLIP未加载**
```bash
# 这是正常现象！采用延迟加载策略
# 首次点击"开始生成"时会自动下载模型
# 需要网络连接到HuggingFace Hub
```

**Q5: 训练中断如何继续？**
```bash
# 直接重新运行 python train.py
# 程序会自动检测断点文件 (*_resume.pt)
# 从上次中断的位置继续训练
```

## 📝 技术细节

### 模型创新点

1. **BERT条件注入**：利用预训练语言模型的语义理解能力，将文本描述转化为连续向量空间
2. **非对称学习率**：判别器backbone使用更低学习率（0.1倍），保持预训练特征提取能力
3. **多归一化对比**：系统性比较BatchNorm vs InstanceNorm在条件生成任务中的效果
4. **双模型协同**：cGAN提供快速原型，SD提供高质量细化，形成互补

### 训练技巧

- **梯度惩罚**：使用Wasserstein损失 + 梯度 penalty稳定训练
- **谱归一化**：判别器最后一层使用谱归一化约束Lipschitz常数
- **EMA平滑**：可选的指数移动平均提升生成质量（可扩展）
- **数据增强**：随机水平翻转、颜色抖动等

## 📄 许可证

本项目仅供学习和研究使用。

## 👨‍💻 作者信息

**课程项目** - 期末作业
- 项目名称：跨模态图像生成系统
- 技术栈：PyTorch + HuggingFace Diffusers + Gradio
- 数据集：Oxford Flowers 102 Category Dataset

## 🙏 致谢

- [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/flowers/) - Flowers102数据集
- [Hugging Face](https://huggingface.co/) - Transformers & Diffusers库
- [Stability AI](https://stability.ai/) - Stable Diffusion v1.5模型
- [Gradio](https://gradio.app/) - Web UI框架
- PyTorch团队 - 深度学习框架

---

**📧 联系方式**：如有问题欢迎提Issue讨论

**⭐ 如果这个项目对你有帮助，请给一个Star支持一下！**
