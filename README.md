# PaDiM + YOLO 異常検知システム

Dockerコンテナ内で、YOLO による人物検出と PaDiM による異常検知を行うシステムです。

## 前提条件

- Docker
- Docker Compose
- NVIDIA GPU（推奨）
- NVIDIA Container Toolkit（GPU使用時）

## セットアップ

### 1. コンテナのビルドと起動

```bash
# ビルドと起動
docker compose up -d

# コンテナに入る
docker compose exec app bash
```

### 2. 使い方

推論時に異常検出閾値を指定できます（デフォルト: 0.5）：

```bash
# コンテナ内で実行
# デフォルト閾値（0.5）で推論
python main.py /app/images/

# カスタム閾値で推論
python main.py /app/images/ --threshold 0.7
```

---

## 推論フェーズ

学習済みモデルを使って画像の異常検知を実行します。

### 単一画像の推論

```bash
# コンテナ内で実行
python main.py /app/images/sample.jpg
```

### フォルダ内の全画像を推論

```bash
# コンテナ内で実行
python main.py /app/images/
```

### 推論結果

推論結果は以下の形式で出力されます：

- **no_person**: 人が検出されなかった
- **normal**: 人が検出され、異常なし
- **anomaly**: 人が検出され、異常あり
- **error**: 処理エラー

結果は`logs/results_YYYYMMDD.jsonl`に記録されます。

---

## 学習フェーズ

PaDiMモデルを学習させるための手順です。

### ステップ1: データセット準備

`images/`フォルダ内の画像から学習用データセットを作成します。

```bash
# コンテナ内で実行
python scripts/prepare_dataset.py
```

**処理内容:**

- `images/`フォルダ内の画像を`dataset/`にコピー
- 画像を224×224にリサイズ
- 学習用のディレクトリ構造に変換

**データセット構造:**

```plaintext
dataset/
├── train/
│   └── good/       # 正常画像（学習用）
└── test/
    ├── good/       # 正常画像（テスト用）
    └── defect/     # 異常画像（テスト用、オプション）
```

### ステップ2: モデル学習

データセット準備後、PaDiMモデルを学習します。

```bash
# コンテナ内で実行
python scripts/train_padim.py
```

**学習設定:**

- バックボーン: ResNet18
- 画像サイズ: 224×224
- 学習済みモデル保存先: `models/padim_trained.ckpt`

### 学習のポイント

1. **正常画像を十分に用意**: 各カテゴリ50枚以上推奨
2. **prepare_dataset.pyを先に実行**: データセット構造を作成してから学習
3. **異常画像はオプション**: テスト用に用意すると精度評価が可能

---

## ディレクトリ構造

```plaintext
.
├── main.py                    # メイン推論スクリプト
├── person_detector.py         # YOLO人物検出モジュール
├── scripts/
│   ├── prepare_dataset.py    # データセット準備スクリプト
│   └── train_padim.py        # PaDiM学習スクリプト
├── .env.example              # 環境変数テンプレート
├── Dockerfile                # Dockerコンテナ設定
├── docker-compose.yml        # Docker Compose設定
├── images/                   # 元画像ディレクトリ（推論対象／学習元）
├── dataset/                  # 学習用データセット（prepare_dataset.pyで生成）
│   ├── train/good/
│   └── test/good/
├── models/                   # モデル格納ディレクトリ
│   ├── yolo11n.pt           # YOLO11モデル（自動ダウンロード）
│   ├── yolo11n.engine       # TensorRT変換後のYOLOモデル
│   └── padim_trained.ckpt   # 学習済みPaDiMモデル
└── logs/                     # ログファイル
    ├── main_YYYYMMDD.log
    ├── prepare_training_data_YYYYMMDDHHMMSS.log
    └── results_YYYYMMDD.jsonl
```

## グリッド分割

画像を 4×4 の 16 分割し、人が検出された位置を特定：

```plaintext
grid_00  grid_01  grid_02  grid_03
grid_04  grid_05  grid_06  grid_07
grid_08  grid_09  grid_10  grid_11
grid_12  grid_13  grid_14  grid_15
```

## 処理フロー

### 推論フロー

1. **画像読み込み** → 指定された画像またはフォルダ内の画像を読み込み
2. **YOLO 検出**
   - 人が写っていない → `no_person`と判定
   - 人が写っている → 3 へ
3. **グリッド位置特定** → 人の位置から grid_XX 番号を取得
4. **PaDiM 異常検知**
   - 異常検出 → `anomaly`と判定
   - 正常 → `normal`と判定
   - エラー → `error`と判定
5. **結果ログ記録** → `logs/results_YYYYMMDD.jsonl` に記録

### 学習フロー

1. **データ準備** → `prepare_dataset.py`実行
   - `images/`から画像を読み込み
   - 224×224にリサイズ
   - `dataset/`にコピー
2. **モデル学習** → `train_padim.py`実行
   - `dataset/train/good/`で学習
   - `dataset/test/`で検証
   - `models/padim_trained.ckpt`に保存

## Docker環境

### コンテナ管理

```bash
# コンテナ起動
docker compose up -d

# コンテナに入る
docker compose exec app bash

# コンテナ停止
docker compose down

# ログ確認
docker compose logs -f app
```

### 環境変数

docker-compose.ymlで以下の環境変数が自動設定されます：

- `USERNAME`: ホストのユーザー名（デフォルト: appuser）
- `USER_UID`: ホストのUID（デフォルト: 1000）
- `USER_GID`: ホストのGID（デフォルト: 1000）

### ビルド引数

Dockerfileのビルド引数は`docker-compose.yml`の`args`セクションで設定：

```yaml
build:
  context: .
  args:
    USERNAME: ${USERNAME:-appuser}
    USER_UID: ${UID:-1000}
    USER_GID: ${GID:-1000}
```

## 設定

### YOLO モデル

- 初回実行時に`models/yolo11n.pt`が自動ダウンロードされます
- TensorRT形式（`.engine`）に変換して高速化されます
- 人（person）クラスの検出に使用

### PaDiM モデル

- 初回推論時は未学習モデルを使用
- 学習フェーズで`scripts/prepare_dataset.py` → `scripts/train_padim.py`の順に実行
- 学習済みモデルは`models/padim_trained.ckpt`に保存

### 異常検出閾値

推論実行時にコマンドライン引数で指定（デフォルト: 0.5）：

```bash
# コンテナ内で実行
# デフォルト閾値（0.5）で推論
python main.py /app/images/

# カスタム閾値で推論
python main.py /app/images/sample.jpg --threshold 0.7
```

## ログ

### 推論ログ

- `logs/main_YYYYMMDD.log`: 推論処理のログ
- `logs/results_YYYYMMDD.jsonl`: JSON形式の処理結果

```json
{
  "timestamp": "20250109_120000",
  "image_path": "/app/images/sample.jpg",
  "person_detected": true,
  "person_info": {
    "grid_index": 5,
    "grid_x": 1,
    "grid_y": 1,
    "confidence": 0.95
  },
  "padim_result": {
    "anomaly_score": 0.12,
    "threshold": 0.5,
    "is_anomaly": false
  },
  "final_decision": "normal"
}
```

### 学習ログ

- `logs/prepare_training_data_YYYYMMDDHHMMSS.log`: データセット準備のログ

## トラブルシューティング

### YOLOモデルの自動ダウンロード

初回実行時に`models/yolo11n.pt`が存在しない場合、自動的にダウンロードされます。

### PaDiM学習がうまくいかない場合

1. **データセットが作成されていない**
   - 先に`python scripts/prepare_dataset.py`を実行してください
2. **学習用画像が少ない**
   - 各カテゴリ50枚以上の画像を用意してください
3. **データセット構造の確認**
   - `dataset/train/good/`に正常画像が配置されているか確認

### ログの確認

```bash
# コンテナ内で実行
# 推論ログを確認
tail -f logs/main_$(date +%Y%m%d).log

# 結果ログを確認
tail -f logs/results_$(date +%Y%m%d).jsonl

# データ準備ログを確認（最新）
ls -t logs/prepare_training_data_*.log | head -1 | xargs tail -f
```

### GPUが認識されない場合

```bash
# コンテナ内でGPU確認
nvidia-smi

# Docker Composeの設定を確認
# docker-compose.ymlでruntime: nvidiaが設定されているか確認
```

## 主な特徴

### 推論フェーズ

- **フォルダ一括処理**: 指定フォルダ内の全画像を一括で推論
- **柔軟な入力**: 単一画像またはフォルダを指定可能
- **詳細なログ**: JSON形式で処理結果を記録

### 学習フェーズ

- **自動データセット準備**: `prepare_dataset.py`で学習用データを自動生成
- **簡単な学習手順**: 2ステップで学習完了
- **モデル自動保存**: 学習済みモデルを自動的に保存

### Docker環境

- **モデル自動ダウンロード**: YOLO モデルの自動取得
- **GPU対応**: NVIDIA GPUによる高速推論
- **権限管理**: ホストユーザーの UID/GID でコンテナ実行（ファイル権限問題なし）
- **自動設定**: 環境変数から自動的にUID/GIDを取得

## Docker環境での権限管理

このシステムは、ホストユーザーと同一の UID/GID で Docker コンテナを実行するため、ファイル権限の問題が発生しません。

### 自動設定の仕組み

docker-compose.ymlの`args`セクションで環境変数からビルド引数を設定：

```yaml
build:
  context: .
  args:
    USERNAME: ${USERNAME:-appuser}
    USER_UID: ${UID:-1000}
    USER_GID: ${GID:-1000}
```

### 利点

- ✅ **ファイル権限問題なし**: ホストとコンテナ間でファイル編集が自由
- ✅ **セキュリティ**: root ではなく一般ユーザーで実行
- ✅ **自動設定**: 環境変数から自動的にUID/GIDを取得
- ✅ **ポータビリティ**: どのLinux環境でも同じように動作
