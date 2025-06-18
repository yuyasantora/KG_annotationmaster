# ベースとなるPython環境を選択
FROM python:3.9-slim

# Streamlitが使用するポートを公開
EXPOSE 8501

# OpenCVの動作に必要なOSライブラリをインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを作成し、移動
WORKDIR /app

# 必要なライブラリをインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトのファイルをコンテナにコピー
COPY . .

# アプリケーションの起動コマンド
# OPENAI_API_KEYは起動時に環境変数として渡すことを推奨
# CMD ["streamlit", "run", "app.py"]
CMD streamlit run app.py --server.port 8501 --server.address 0.0.0.0
