#!/usr/bin/env python3
"""
PaDiM + YOLO 異常検知メインスクリプト

指定フォルダ内の画像で推論を実行:
1. 指定フォルダ内の画像を読み込み
2. YOLO で人の検出
3. PaDiM で異常検知
4. 結果の記録と画像保存
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import numpy as np

from anomalib.models import Padim
from anomalib.engine import Engine
from dotenv import load_dotenv
from person_detector import detect_person_and_get_grid


# 環境変数を明示的に読み込み
try:
    load_dotenv()
except ImportError:
    pass


def setup_logging():
    """ログ設定"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_filename = f"main_{datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


class PaDiMAnomalyDetector:
    """PaDiM 異常検知クラス"""

    def __init__(self, model_path: str = "models/padim_trained.ckpt", anomaly_threshold: float = 0.5):
        self.model_path = model_path
        self.anomaly_threshold = anomaly_threshold
        self.model = None
        self.engine = None
        self._load_model()

    def _load_model(self):
        """PaDiMモデルの読み込み"""
        try:
            if os.path.exists(self.model_path):
                # 学習済みモデルがある場合
                try:
                    # 新しい形式（.save()で保存されたモデル）を試行
                    self.model = Padim.load(self.model_path)
                    self.model.eval()
                    logging.info(
                        f"学習済みPaDiMモデルを読み込みました（新形式）: {self.model_path}"
                    )
                except Exception:
                    try:
                        # 古い形式（チェックポイント）を試行
                        self.model = Padim.load_from_checkpoint(self.model_path)
                        self.model.eval()
                        logging.info(
                            f"学習済みPaDiMモデルを読み込みました（チェックポイント）: {self.model_path}"
                        )
                    except Exception as e:
                        logging.warning(f"モデル読み込みに失敗、初期モデルを使用: {e}")
                        self._create_initial_model()
            else:
                logging.warning(
                    "学習済みモデルが見つかりません。初期モデルを使用します。"
                )
                self._create_initial_model()

            self.engine = Engine()

        except Exception as e:
            logging.error(f"PaDiMモデルの読み込みに失敗: {e}")
            raise

    def _create_initial_model(self):
        """初期モデルの作成"""
        pre_processor = Padim.configure_pre_processor(image_size=(224, 224))
        self.model = Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=True,
            pre_processor=pre_processor,
        )

    def predict(self, image_path: str) -> Dict[str, Any]:
        """異常検知の実行（推論専用）"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

            # 直接画像を読み込んで推論
            import torch
            from torchvision import transforms
            from PIL import Image

            # 画像を読み込み
            image = Image.open(image_path).convert("RGB")
            
            # 前処理（PaDiMの標準前処理）
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加
            
            # GPUが利用可能な場合はGPUを使用
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                if hasattr(self.model, 'cuda'):
                    self.model = self.model.cuda()
            
            # 推論モードに設定
            self.model.eval()
            
            with torch.no_grad():
                # 直接モデルで推論
                output = self.model(image_tensor)
                
                # anomalib v0.7.0以降の場合
                if hasattr(output, 'pred_score'):
                    anomaly_score = float(output.pred_score.cpu().item())
                    anomaly_map = output.anomaly_map.cpu().numpy() if hasattr(output, 'anomaly_map') else None
                # 辞書形式の場合
                elif isinstance(output, dict):
                    anomaly_score = float(output.get('pred_score', 0.0))
                    anomaly_map = output.get('anomaly_map', None)
                    if anomaly_map is not None and hasattr(anomaly_map, 'cpu'):
                        anomaly_map = anomaly_map.cpu().numpy()
                # フォールバック: 出力テンソルから直接スコアを取得
                else:
                    if hasattr(output, 'cpu'):
                        anomaly_score = float(torch.mean(output).cpu().item())
                        anomaly_map = output.cpu().numpy()
                    else:
                        anomaly_score = float(torch.mean(torch.tensor(output)).item())
                        anomaly_map = output
                
                result = {
                    "anomaly_score": anomaly_score,
                    "is_anomaly": anomaly_score > self.anomaly_threshold,
                    "anomaly_map": anomaly_map,
                    "threshold": self.anomaly_threshold,
                }

                # 最も異常度が高い座標区画を特定
                if anomaly_map is not None:
                    max_anomaly_coords = self._find_max_anomaly_coordinates(
                        anomaly_map
                    )
                    result["max_anomaly_coordinates"] = max_anomaly_coords

                logging.info(
                    f"PaDiM推論完了: score={result['anomaly_score']:.4f}, threshold={self.anomaly_threshold:.4f}, anomaly={result['is_anomaly']}"
                )
                return result

        except Exception as e:
            logging.error(f"PaDiM異常検知エラー: {e}")
            return {
                "anomaly_score": 0.0,
                "is_anomaly": False,
                "anomaly_map": None,
                "max_anomaly_coordinates": None,
                "error": str(e),
            }

    def _find_max_anomaly_coordinates(self, anomaly_map: np.ndarray) -> Dict[str, Any]:
        """異常マップから最も異常度が高い座標区画を特定"""
        try:
            if anomaly_map.ndim == 3:
                # チャンネル次元がある場合は最初のチャンネルを使用
                anomaly_map = anomaly_map[0]
            elif anomaly_map.ndim == 4:
                # バッチ次元とチャンネル次元がある場合
                anomaly_map = anomaly_map[0, 0]

            # 最大異常値とその座標を取得
            max_anomaly_value = float(np.max(anomaly_map))
            max_coords = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)

            # 座標を画像サイズに正規化（0-1の範囲）
            height, width = anomaly_map.shape
            normalized_y = float(max_coords[0]) / height
            normalized_x = float(max_coords[1]) / width

            # 4x4グリッドでの区画番号を計算
            grid_y = int(normalized_y * 4)
            grid_x = int(normalized_x * 4)
            grid_index = grid_y * 4 + grid_x

            # 範囲チェック
            grid_y = min(max(grid_y, 0), 3)
            grid_x = min(max(grid_x, 0), 3)
            grid_index = min(max(grid_index, 0), 15)

            return {
                "max_anomaly_value": max_anomaly_value,
                "pixel_coordinates": {"x": int(max_coords[1]), "y": int(max_coords[0])},
                "normalized_coordinates": {"x": normalized_x, "y": normalized_y},
                "grid_coordinates": {"x": grid_x, "y": grid_y},
                "grid_index": grid_index,
                "anomaly_map_shape": list(anomaly_map.shape),
            }

        except Exception as e:
            logging.error(f"最大異常座標の特定でエラー: {e}")
            return {
                "error": str(e),
                "max_anomaly_value": 0.0,
                "pixel_coordinates": {"x": 0, "y": 0},
                "normalized_coordinates": {"x": 0.0, "y": 0.0},
                "grid_coordinates": {"x": 0, "y": 0},
                "grid_index": 0,
                "anomaly_map_shape": [],
            }


class MainProcessor:
    def process_image_file(self, image_path: str) -> Dict[str, Any]:
        """指定画像ファイルで推論を実行"""
        try:
            person_result = detect_person_and_get_grid(image_path, self.yolo_model)
            result = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "image_path": image_path,
                "person_detected": person_result is not None,
                "person_info": person_result,
                "padim_result": None,
                "final_decision": None,
            }

            if person_result is None:
                result["final_decision"] = "no_person"
            else:
                padim_result = self.padim_detector.predict(image_path)
                result["padim_result"] = padim_result

                if padim_result.get("error"):
                    result["final_decision"] = "error"
                else:
                    if padim_result.get("is_anomaly", False):
                        result["final_decision"] = "anomaly"
                    else:
                        result["final_decision"] = "normal"

            self._save_result_log(result)
            return result
        except Exception as e:
            self.logger.error(f"画像ファイル処理中にエラーが発生: {e}")
            return {"error": str(e), "image_path": image_path}
    """メイン処理クラス"""

    def __init__(self, anomaly_threshold: float = None):
        self.logger = setup_logging()

        
        # 閾値設定（環境変数またはパラメータから取得）
        if anomaly_threshold is None:
            anomaly_threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
        
        self.padim_detector = PaDiMAnomalyDetector(anomaly_threshold=anomaly_threshold)
        self.yolo_model = None
        self._load_yolo_model()
        
        self.logger.info(f"異常検出閾値を設定: {anomaly_threshold}")

    def _load_yolo_model(self):
        """YOLOモデルの読み込み"""
        try:
            from ultralytics import YOLO

            # modelsディレクトリを作成
            os.makedirs("models", exist_ok=True)

            if not os.path.exists("models/yolo11n.engine"):
                self.logger.info("yolo11n.engineファイルが見つかりません。")
                # なければYOLOモデルを自動ダウンロード
                self.yolo_model = YOLO("models/yolo11n.pt")
                # TensorRTエンジンを生成
                self.yolo_model.export(format="engine", task="detect")
            self.logger.info("yolo11n.engineファイルを使用")
            self.yolo_model = YOLO("models/yolo11n.engine", task="detect")
            self.logger.info("YOLOモデルを読み込みました")

        except Exception as e:
            self.logger.error(f"YOLOモデルの読み込みに失敗: {e}")
            raise

    def process_images_in_folder(self, folder_path: str) -> list[Dict[str, Any]]:
        """指定フォルダ内の全画像で推論を実行"""
        results = []
        folder = Path(folder_path)

        if not folder.exists():
            self.logger.error(f"フォルダが存在しません: {folder_path}")
            return results

        # 画像ファイルの拡張子
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # フォルダ内の画像ファイルを取得
        image_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            self.logger.warning(f"フォルダ内に画像ファイルが見つかりません: {folder_path}")
            return results

        self.logger.info(f"フォルダ内の画像数: {len(image_files)}")

        for image_file in image_files:
            self.logger.info(f"処理中: {image_file}")
            result = self.process_image_file(str(image_file))
            results.append(result)

        return results

    def _save_result_log(self, result: Dict[str, Any]):
        """結果をJSONログとして保存"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"results_{datetime.now().strftime('%Y%m%d')}.jsonl"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, default=str)
                f.write("\n")
        except Exception as e:
            self.logger.error(f"結果ログの保存に失敗: {e}")


def main():
    """メイン関数"""
    try:
        import argparse

        # コマンドライン引数のパーサー設定
        parser = argparse.ArgumentParser(description='PaDiM + YOLO 異常検知システム')
        parser.add_argument('path', help='画像ファイルまたはフォルダのパス')
        parser.add_argument('--threshold', type=float, default=0.5,
                          help='異常検出閾値 (デフォルト: 0.5)')

        args = parser.parse_args()

        # 閾値を指定してプロセッサを初期化
        processor = MainProcessor(anomaly_threshold=args.threshold)

        path = args.path
        path_obj = Path(path)

        # パスがディレクトリかファイルかで処理を分岐
        if path_obj.is_dir():
            # フォルダ内の全画像を処理
            results = processor.process_images_in_folder(path)

            if not results:
                print("処理する画像がありませんでした")
                return 1

            # 結果のサマリーを表示
            print("\n=== 処理完了 ===")
            print(f"処理画像数: {len(results)}")

            # 判定別の集計
            summary = {
                'no_person': 0,
                'normal': 0,
                'anomaly': 0,
                'error': 0
            }

            for result in results:
                decision = result.get('final_decision', 'error')
                if decision in summary:
                    summary[decision] += 1
                else:
                    summary['error'] += 1

            print(f"人検出なし: {summary['no_person']}")
            print(f"正常: {summary['normal']}")
            print(f"異常: {summary['anomaly']}")
            print(f"エラー: {summary['error']}")

        elif path_obj.is_file():
            # 単一画像を処理
            result = processor.process_image_file(path)

            print("\n=== 処理結果 ===")
            print(f"タイムスタンプ: {result.get('timestamp', 'N/A')}")
            print(f"人検出: {result.get('person_detected', False)}")

            if result.get("person_info"):
                person_info = result["person_info"]
                print(
                    f"検出位置: Grid {person_info['grid_index']:02d} (x={person_info['grid_x']}, y={person_info['grid_y']})"
                )
                print(f"信頼度: {person_info['confidence']:.4f}")

            if result.get("padim_result"):
                padim_result = result["padim_result"]
                print(f"異常スコア: {padim_result.get('anomaly_score', 0):.4f}")
                print(f"異常閾値: {padim_result.get('threshold', 0):.4f}")
                print(f"異常判定: {padim_result.get('is_anomaly', False)}")

                # 最も異常度が高い座標区画の情報を出力
                max_anomaly_coords = padim_result.get("max_anomaly_coordinates")
                if max_anomaly_coords and not max_anomaly_coords.get("error"):
                    print("=== 最も異常度が高い座標区画 ===")
                    print(f"最大異常値: {max_anomaly_coords['max_anomaly_value']:.6f}")
                    print(
                        f"ピクセル座標: x={max_anomaly_coords['pixel_coordinates']['x']}, y={max_anomaly_coords['pixel_coordinates']['y']}"
                    )
                    print(
                        f"正規化座標: x={max_anomaly_coords['normalized_coordinates']['x']:.4f}, y={max_anomaly_coords['normalized_coordinates']['y']:.4f}"
                    )
                    print(
                        f"グリッド座標: x={max_anomaly_coords['grid_coordinates']['x']}, y={max_anomaly_coords['grid_coordinates']['y']}"
                    )
                    print(f"グリッド番号: Grid {max_anomaly_coords['grid_index']:02d}")
                    print(f"異常マップサイズ: {max_anomaly_coords['anomaly_map_shape']}")

            print(f"最終判定: {result.get('final_decision', 'unknown')}")

            if result.get("error"):
                print(f"エラー: {result['error']}")
                return 1
        else:
            print(f"エラー: パスが見つかりません: {path}")
            return 1

        return 0

    except Exception as e:
        print(f"メイン処理でエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
