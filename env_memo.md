🐍 方法1: Conda環境の構築（推奨）
# Conda環境の作成
conda create -n transformer_brain python=3.9
conda activate transformer_brain

# 基本パッケージのインストール
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers
pip install open_clip_torch
pip install scikit-learn scipy nilearn
pip install tqdm matplotlib pillow

🐍 方法2: venv環境の構築
# 仮想環境の作成
python -m venv transformer_brain_env

# 仮想環境の有効化
# Windows:
transformer_brain_env\Scripts\activate
# Linux/Mac:
source transformer_brain_env/bin/activate

# 依存パッケージのインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install open_clip_torch
pip install scikit-learn scipy nilearn
pip install tqdm matplotlib pillow