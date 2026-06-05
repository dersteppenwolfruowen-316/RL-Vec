"""RL Vectorizer — Colab 一键 SFT 训练脚本"""
import os, sys, urllib.request, zipfile, subprocess

# 1. 克隆仓库（如果没克隆过）
REPO = "/content/RL-Vec"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "https://github.com/dersteppenwolfruowen-316/RL-Vec.git"], check=True)

os.chdir(REPO)

# 2. 装依赖
os.system("pip install -q torch transformers peft accelerate bitsandbytes datasets")
os.system("pip install -q lxml cairosvg pillow scikit-image opencv-python tensorboard hydra-core omegaconf shapely")

# 3. 下载 ResPlan 数据
DATA_DIR = "data/resplan"
os.makedirs(DATA_DIR, exist_ok=True)
ZIP_PATH = os.path.join(DATA_DIR, "ResPlan.zip")

if not os.path.exists(ZIP_PATH):
    print("Downloading ResPlan.zip (96MB)...")
    urllib.request.urlretrieve(
        "https://github.com/m-agour/ResPlan/releases/download/1.0.0/ResPlan.zip",
        ZIP_PATH)
    print("Download complete!")

if not os.path.exists(os.path.join(DATA_DIR, "ResPlan.pkl")):
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(DATA_DIR)
    print("Extracted!")

if not os.path.exists(os.path.join(DATA_DIR, "svgs")):
    subprocess.run(["python", "convert_resplan.py"], check=True)

if not os.path.exists(os.path.join(DATA_DIR, "sft_train.jsonl")):
    subprocess.run(["python", "scripts/prepare_sft_data.py"], check=True)

print("Data ready!")
print(f"SFT data: {os.path.getsize(os.path.join(DATA_DIR, 'sft_train.jsonl'))/1024/1024:.1f}MB")

# 4. SFT 训练
print("\n===== Starting SFT Training =====")
subprocess.run([
    "python", "scripts/train_sft.py",
    "--max-samples", "200",
    "--batch-size", "1",
    "--epochs", "3",
    "--lr", "1e-4",
    "--save-dir", "checkpoints/sft",
    "--log-interval", "5",
], check=True)

print("\n===== SFT Training Complete! =====")
