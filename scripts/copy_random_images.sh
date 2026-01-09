#!/bin/bash
# ランダム画像コピー高速スクリプト
# copy_images.pyの完全置き換え

set -e

# デフォルト値設定
DEFAULT_SRC_DIR="/source/path"
DEFAULT_DST_DIR="/destination/path"
DEFAULT_IMAGES_PER_FOLDER=200

SRC_DIR="${1:-$DEFAULT_SRC_DIR}"
DST_DIR="${2:-$DEFAULT_DST_DIR}"
IMAGES_PER_FOLDER="${3:-$DEFAULT_IMAGES_PER_FOLDER}"

# 使用法表示関数
show_usage() {
    echo "使用法: $0 [source_dir] [destination_dir] [images_per_folder]"
    echo ""
    echo "引数:"
    echo "  source_dir        ソースディレクトリのパス (デフォルト: $DEFAULT_SRC_DIR)" 
    echo "  destination_dir   出力先ディレクトリのパス (デフォルト: $DEFAULT_DST_DIR)"
    echo "  images_per_folder 各フォルダからコピーする画像数 (デフォルト: $DEFAULT_IMAGES_PER_FOLDER)"
    echo ""
    echo "例:"
    echo "  $0 /path/to/source /path/to/dest 150"
    echo "  $0 # デフォルト値を使用"
}

# ヘルプオプション
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

echo "ソースディレクトリ: $SRC_DIR"
echo "出力先ディレクトリ: $DST_DIR"
echo "各フォルダからコピーする画像数: $IMAGES_PER_FOLDER"
echo ""

# ソースディレクトリの確認
if [ ! -d "$SRC_DIR" ]; then
    echo "エラー: ソースディレクトリが存在しません: $SRC_DIR"
    echo ""
    echo "現在のディレクトリ: $(pwd)"
    echo "利用可能なディレクトリ:"
    ls -la . 2>/dev/null || echo "  (ディレクトリ一覧を取得できませんでした)"
    echo ""
    show_usage
    exit 1
fi

# 出力先ディレクトリが存在しない場合は作成
mkdir -p "$DST_DIR"

# 各フォルダを処理
for folder in "$SRC_DIR"/*; do
    if [ ! -d "$folder" ]; then
        continue
    fi
    
    folder_name=$(basename "$folder")
    echo "処理中のフォルダ: $folder_name"
    
    # 画像ファイルを一時ファイルに収集
    temp_file=$(mktemp)
    
    # 画像ファイルをリストアップ
    find "$folder" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.tif" -o -iname "*.webp" \) > "$temp_file"
    
    image_count=$(wc -l < "$temp_file")
    echo "  - 見つかった画像ファイル数: $image_count"
    
    if [ "$image_count" -eq 0 ]; then
        echo "  - スキップ: 画像ファイルがありません"
        rm -f "$temp_file"
        continue
    fi
    
    # 出力先フォルダを作成
    dst_folder="$DST_DIR/$folder_name"
    mkdir -p "$dst_folder"
    
    # ランダムに選択する枚数を決定
    copy_count=$((image_count < IMAGES_PER_FOLDER ? image_count : IMAGES_PER_FOLDER))
    echo "  - コピーする画像数: $copy_count"
    
    # ランダムに選択してコピー
    copied_count=0
    shuf "$temp_file" | head -n "$copy_count" | while read -r img_file; do
        if [ -f "$img_file" ]; then
            img_name=$(basename "$img_file")
            dst_file="$dst_folder/$img_name"
            if cp "$img_file" "$dst_file" 2>/dev/null; then
                ((copied_count++)) || true
            else
                echo "  - エラー: $img_name のコピーに失敗しました"
            fi
        fi
    done
    
    # 実際にコピーされた数を確認
    actual_copied=$(find "$dst_folder" -type f | wc -l)
    echo "  - 完了: ${actual_copied}枚の画像をコピーしました"
    
    rm -f "$temp_file"
done

echo ""
echo "すべての処理が完了しました！"