import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from anomalib.data import datamodules, Folder
from anomalib.engine import Engine
from anomalib.models import Padim


def setup_logging() -> logging.Logger:
    """ログ設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_filename = f"train_padim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def count_images_in_directory(directory: Path) -> int:
    """ディレクトリ内の画像ファイル数をカウント"""
    if not directory.exists():
        return 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    count = 0

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            count += 1

    return count


def create_padim_model(
    image_size: tuple = (224, 224),  # ResNet標準サイズ（最適な処理効率）
    backbone: str = "resnet18",
    layers: List[str] | None = None,
) -> Padim:
    """PaDiMモデルの作成"""
    if layers is None:
        layers = ["layer1", "layer2", "layer3"]
    pre_processor = Padim.configure_pre_processor(image_size=image_size)

    model = Padim(
        backbone=backbone,
        layers=layers,
        pre_trained=True,
        pre_processor=pre_processor,
    )

    return model


def train_test_padim_model(
    datamodule: datamodules,
    image_size: tuple = (224, 224),  # ResNet標準サイズ（最適な処理効率）
    batch_size: int = 32,
    num_workers: int = 2,
) -> None:
    """PaDiMモデルの学習"""
    logger = logging.getLogger(__name__)

    logger.info("PaDiMモデルの学習を開始します")
    logger.info("元画像サイズ: 640x480 (カメラ解像度)")
    logger.info(f"リサイズ後サイズ: {image_size} (処理効率のため)")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"ワーカー数: {num_workers}")

    # データモジュールをセットアップ
    logger.info("データモジュールをセットアップ中...")
    datamodule.setup()
    logger.info("データモジュールのセットアップが完了しました")

    # データモジュールの詳細情報を表示
    logger.info("=" * 40)
    logger.info("データモジュール内訳")
    logger.info("=" * 40)

    try:
        if hasattr(datamodule, "train_dataset") and datamodule.train_dataset:
            train_size = len(datamodule.train_dataset)
            logger.info(f"学習用データセット: {train_size} 枚")

        if hasattr(datamodule, "val_dataset") and datamodule.val_dataset:
            val_size = len(datamodule.val_dataset)
            logger.info(f"検証用データセット: {val_size} 枚")

        if hasattr(datamodule, "test_dataset") and datamodule.test_dataset:
            test_size = len(datamodule.test_dataset)
            logger.info(f"テスト用データセット: {test_size} 枚")

        # データローダーの情報
        if hasattr(datamodule, "train_dataloader"):
            train_loader = datamodule.train_dataloader()
            if train_loader:
                logger.info(f"学習用バッチ数: {len(train_loader)} バッチ")

        if hasattr(datamodule, "val_dataloader"):
            val_loader = datamodule.val_dataloader()
            if val_loader:
                logger.info(f"検証用バッチ数: {len(val_loader)} バッチ")

        if hasattr(datamodule, "test_dataloader"):
            test_loader = datamodule.test_dataloader()
            if test_loader:
                logger.info(f"テスト用バッチ数: {len(test_loader)} バッチ")

    except Exception as e:
        logger.warning(f"データモジュール情報の取得に失敗: {e}")
        raise

    logger.info("=" * 40)

    # モデルの準備（画像サイズを明示的に指定）
    logger.info(f"PaDiMモデルを作成中（画像サイズ: {image_size}）")
    model = create_padim_model(image_size=image_size, backbone="resnet18")
    engine = Engine()

    # 学習実行
    logger.info("=" * 50)
    logger.info("PaDiMモデル学習開始")
    logger.info("=" * 50)
    logger.info(f"学習開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("モデルバックボーン: resnet18")
    logger.info("特徴抽出レイヤー: ['layer1', 'layer2', 'layer3']")
    logger.info("=" * 50)

    try:
        engine.fit(model=model, datamodule=datamodule)
        logger.info("=" * 50)
        logger.info("学習が正常に完了しました")
        logger.info(f"学習完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
    except Exception as e:
        logger.error("=" * 50)
        logger.error("学習中にエラーが発生しました")
        logger.error(f"エラー詳細: {e}")
        logger.error("=" * 50)
        raise

    logger.info("=" * 30)
    logger.info("テスト開始")
    logger.info("=" * 30)

    try:
        # testを実行
        logger.info("テスト実行中...")
        test_results = engine.test(model=model, datamodule=datamodule)

        logger.info("=" * 30)
        logger.info("テスト完了")
        logger.info(f"テスト結果: {test_results}")
        logger.info("=" * 30)

    except Exception as e:
        logger.error(f"テスト実行中にエラーが発生: {e}")
        logger.warning("テストに失敗しましたが、学習は正常に完了しています")

    logger.info("🎉 PaDiMモデル学習が正常に完了しました 🎉")


def main():
    dataset_root = "./dataset"
    image_size = (224, 224)
    batch_size = 128
    num_workers = 2

    # ログ設定
    logger = setup_logging()

    # Folderデータモジュールを使用（dataset/good, dataset/defect）
    datamodule = Folder(
        name="padim_train",
        root=".",
        normal_dir="dataset/good",
        abnormal_dir="dataset/defect",
        train_batch_size=batch_size,
        num_workers=num_workers,
        val_split_mode="from_test",
        test_split_mode="from_dir",
    )
    logger.info(
        f"Folderデータモジュールを作成しました (batch_size={batch_size}, num_workers={num_workers})"
    )

    # モデル学習の実行
    train_test_padim_model(
        datamodule=datamodule,
        image_size=tuple(image_size),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logger.info("すべての処理が完了しました")
    return


if __name__ == "__main__":
    main()
