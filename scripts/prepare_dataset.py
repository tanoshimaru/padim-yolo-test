#!/usr/bin/env python3
"""
PaDiM学習用データ準備スクリプト

images/ ディレクトリ内の画像を dataset/ に分散配置し、
PaDiMの学習データセットを準備します。
"""

import sys
import logging
from datetime import datetime
from pathlib import Path


def setup_logging():
    """ログ設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_filename = (
        f"prepare_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)



from PIL import Image

class TrainingDataPreparer:
    """学習データ準備クラス"""

    def __init__(
        self,
        source_dir: str = "./images",
        target_dir: str = "./dataset",
        copy_mode: bool = True,
        random_seed: int = 42,
        resize_size: tuple = (224, 224),
        max_images_per_folder: int | None = None,
    ):
        self.logger = setup_logging()
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.good_dir = self.target_dir / "good"
        self.defect_dir = self.target_dir / "defect"
        self.copy_mode = copy_mode  # True: コピー, False: 移動
        self.random_seed = random_seed
        self.resize_size = resize_size
        self.max_images_per_folder = max_images_per_folder


    def prepare_training_data(self) -> bool:
        """images/defect→dataset/defect, images/defect以外→dataset/goodにリサイズして分類"""
        import shutil
        import random

        try:
            self.logger.info("=== 学習データ準備開始 ===")
            random.seed(self.random_seed)

            # dataset/good, dataset/defect ディレクトリを作成
            self.good_dir.mkdir(parents=True, exist_ok=True)
            self.defect_dir.mkdir(parents=True, exist_ok=True)

            folder_stats = {}  # フォルダごとの統計

            # defect画像
            defect_src = self.source_dir / "defect"
            defect_count = 0
            if defect_src.exists():
                defect_images = [p for p in defect_src.glob("**/*") if p.is_file()]
                if self.max_images_per_folder and len(defect_images) > self.max_images_per_folder:
                    defect_images = random.sample(defect_images, self.max_images_per_folder)
                    self.logger.info(f"defect画像を{len(defect_images)}枚にサンプリング")
                
                for img_path in defect_images:
                    dst_path = self.defect_dir / img_path.name
                    self._resize_and_save(img_path, dst_path)
                    defect_count += 1
                
                folder_stats["defect"] = defect_count
            else:
                self.logger.warning(f"defectディレクトリが存在しません: {defect_src}")

            # good画像(images/defect以外のサブディレクトリ配下)
            good_count = 0
            for subdir in sorted(self.source_dir.iterdir()):
                if subdir.is_dir() and subdir.name != "defect":
                    subdir_images = [p for p in subdir.glob("**/*") if p.is_file()]
                    if self.max_images_per_folder and len(subdir_images) > self.max_images_per_folder:
                        subdir_images = random.sample(subdir_images, self.max_images_per_folder)
                        self.logger.info(f"{subdir.name}画像を{len(subdir_images)}枚にサンプリング")
                    
                    subdir_count = 0
                    for img_path in subdir_images:
                        dst_path = self.good_dir / f"{subdir.name}_{img_path.name}"
                        self._resize_and_save(img_path, dst_path)
                        subdir_count += 1
                    
                    folder_stats[subdir.name] = subdir_count
                    good_count += subdir_count

            self.create_dataset_info(good_count, defect_count, folder_stats)
            self.logger.info(f"defect画像 {defect_count} 枚, good画像 {good_count} 枚を準備しました")
            self.logger.info("=== 学習データ準備完了 ===")
            return True
        except Exception as e:
            self.logger.error(f"学習データ準備でエラー: {e}")
            raise

    def _resize_and_save(self, src_path, dst_path):
        """画像をリサイズして保存"""
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.resize_size, Image.LANCZOS)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(dst_path)
        except Exception as e:
            self.logger.error(f"画像リサイズ保存失敗: {src_path} → {dst_path}: {e}")

    def create_dataset_info(self, train_count: int, val_count: int, folder_stats: dict):
        """データセット情報ファイルを作成"""
        
        # フォルダごとの詳細統計を作成
        folder_details = ""
        for folder_name, count in sorted(folder_stats.items()):
            folder_details += f"  - {folder_name}: {count} 枚\n"
        
        info_content = f"""# PaDiM学習データセット情報

## 作成日時
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## データ統計
- good画像: {train_count} 枚
- defect画像: {val_count} 枚
- 総画像数: {train_count + val_count} 枚

## フォルダごとの枚数
{folder_details}
## ソース設定
- ソースディレクトリ: {self.source_dir}
- goodディレクトリ: {self.good_dir}
- defectディレクトリ: {self.defect_dir}
- 操作モード: {"コピー" if self.copy_mode else "移動"}
- ランダムシード: {self.random_seed}
- 最大画像枚数/フォルダ: {self.max_images_per_folder if self.max_images_per_folder else "制限なし"}
"""

        info_path = self.target_dir / "dataset_info.md"
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(info_content)

        self.logger.info(f"データセット情報を作成: {info_path}")

    def clean_target_directory(self):
        """ターゲットディレクトリをクリーン"""
        import shutil
        if self.target_dir.exists():
            self.logger.info(f"既存のターゲットディレクトリを削除: {self.target_dir}")
            shutil.rmtree(self.target_dir)



def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="PaDiM学習用データセット準備")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="./images",
        help="ソース画像ディレクトリ (default: ./images)",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./dataset",
        help="データセット出力ディレクトリ (default: ./dataset)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="ファイルを移動(デフォルトはコピー)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="ランダムシード (default: 42)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="ターゲットディレクトリを事前にクリーン",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[224, 224],
        help="リサイズ後の画像サイズ (default: 224 224)",
    )
    parser.add_argument(
        "--max_images_per_folder",
        type=int,
        default=None,
        help="各フォルダから選択する最大画像枚数 (指定なしの場合は全画像を使用)",
    )

    args = parser.parse_args()

    try:
        preparer = TrainingDataPreparer(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            copy_mode=not args.move,
            random_seed=args.random_seed,
            max_images_per_folder=args.max_images_per_folder,
        )
        preparer.resize_size = tuple(args.resize)

        if args.clean:
            preparer.clean_target_directory()

        success = preparer.prepare_training_data()

        if success:
            print(f"\n学習データの準備が完了しました: {args.target_dir}")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n処理が中断されました")
        return 1
    except Exception as e:
        print(f"エラーが発生: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
