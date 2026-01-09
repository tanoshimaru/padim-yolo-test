from ultralytics import YOLO
import cv2
import os
from typing import Tuple, Optional, Dict, Any


def get_grid_position(
    center_x: float, center_y: float, image_width: int, image_height: int
) -> Tuple[int, int, int]:
    """画像を4x4の16分割した時の位置を取得

    Args:
        center_x: 人の中央X座標
        center_y: 人の中央Y座標
        image_width: 画像の幅
        image_height: 画像の高さ

    Returns:
        Tuple[int, int, int]: (grid_index, grid_x, grid_y)
            grid_index: 0-15の区画インデックス
            grid_x: 0-3のX方向の区画位置
            grid_y: 0-3のY方向の区画位置
    """
    # 各グリッドのサイズ
    grid_width = image_width / 4
    grid_height = image_height / 4

    # グリッドの位置を計算 (0-3の範囲)
    grid_x = min(int(center_x / grid_width), 3)
    grid_y = min(int(center_y / grid_height), 3)

    # 16分割のインデックス (0-15)
    grid_index = grid_y * 4 + grid_x

    return grid_index, grid_x, grid_y


def detect_person_and_get_grid(
    image_path: str, model: Any = None
) -> Optional[Dict[str, Any]]:
    """画像から人を検出し、区画情報を返す関数

    Args:
        image_path: 画像ファイルのパス
        model: YOLOモデル（Noneの場合は新規作成）

    Returns:
        Optional[Dict[str, Any]]: 人が検出された場合の情報
            {
                'grid_index': int,      # 0-15の区画インデックス
                'grid_x': int,          # 0-3のX方向の区画位置
                'grid_y': int,          # 0-3のY方向の区画位置
                'center_x': float,      # 人の中央X座標
                'center_y': float,      # 人の中央Y座標
                'confidence': float,    # 検出信頼度
                'bbox': Tuple[float, float, float, float],  # バウンディングボックス (x1, y1, x2, y2)
                'person_count': int     # 検出された人数
            }
            人が検出されなかった場合はNone
    """
    # モデルが指定されていない場合は作成
    if model is None:
        if not os.path.exists("models/yolo11n.engine"):
            # なければYOLOモデルを自動ダウンロード
            model = YOLO("models/yolo11n.pt")
            # TensorRTエンジンを生成
            model.export(format="engine", task="detect")
        model = YOLO("models/yolo11n.engine", task="detect")

    # 画像が存在するかチェック
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

    # 画像のサイズを取得
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")

    image_height, image_width = image.shape[:2]

    # 推論実行
    results = model(image_path)
    result = results[0]

    # 人の検出結果を解析
    person_detections = []

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]

            if class_name.lower() == "person":
                # バウンディングボックスの座標を取得
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])

                # 中央座標を計算
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # グリッド位置を取得
                grid_index, grid_x, grid_y = get_grid_position(
                    center_x, center_y, image_width, image_height
                )

                person_detections.append(
                    {
                        "grid_index": grid_index,
                        "grid_x": grid_x,
                        "grid_y": grid_y,
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "confidence": confidence,
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    }
                )

    # 人が検出されなかった場合
    if not person_detections:
        return None

    # 最も信頼度の高い検出結果を選択
    best_detection = max(person_detections, key=lambda x: x["confidence"])
    best_detection["person_count"] = len(person_detections)

    return best_detection
