# Streamlit SQLite アプリ

## WSL2サーバPC環境構築

### 1. WSL2でのDocker環境構築
```bash
# Docker Desktop for Windowsをインストール後、WSL2統合を有効化
# または WSL2内でDocker Engineを直接インストール
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
sudo service docker start
```

### 2. プロジェクトの配置
```bash
# Windowsからファイルをコピー
cp -r /mnt/c/path/to/streamlit_supply ~/
# または git clone
git clone <repository-url> ~/streamlit_supply
cd ~/streamlit_supply
```

### 3. WSL2での起動
```bash
# バックグラウンドで起動
docker compose up -d --build

# ログ確認
docker compose logs -f

# 停止
docker compose down
```

### 4. Windows側からのアクセス設定
```bash
# WSL2のIPアドレス確認
hostname -I

# Windowsファイアウォール設定（PowerShell管理者権限）
# New-NetFirewallRule -DisplayName "WSL2 Streamlit" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow
```

## 開発環境での起動方法

### Docker Compose使用（推奨）
```bash
docker compose up --build
```

### ローカル実行
```bash
uv sync
uv run streamlit run app.py
```

## アクセス
- WSL2内: http://localhost:8501
- Windows側: http://localhost:8501 (Docker Desktop使用時)
- 他PC: http://WSL2_IP:8501

## 機能
- データ登録
- データ一覧表示
- SQLiteデータベース使用