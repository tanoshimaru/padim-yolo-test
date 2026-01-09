from ultralytics import YOLO
import cv2
import os
import shutil
from typing import Dict, List, Tuple
from person_detector import detect_person_and_get_grid


def setup_directories(input_dir: str) -> Tuple[Dict[int, str], str, str]:
    """ディレクトリの作成と設定

    Args:
        input_dir: 入力ディレクトリのパス

    Returns:
        Tuple[Dict[int, str], str, str]: (grid_dirs, no_person_dir, output_dir)
    """
    # 人が検出されなかった画像の振り分け先ディレクトリ
    no_person_dir = os.path.join(input_dir, "no-person")

    # 16分割のグリッド位置別ディレクトリを作成
    grid_dirs = {}
    for i in range(16):
        grid_dir = os.path.join(input_dir, f"grid_{i:02d}")
        grid_dirs[i] = grid_dir
        if not os.path.exists(grid_dir):
            os.makedirs(grid_dir)

    # no-personディレクトリも作成
    if not os.path.exists(no_person_dir):
        os.makedirs(no_person_dir)

    # 出力ディレクトリがなければ作成
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return grid_dirs, no_person_dir, output_dir


def distribute_image(
    image_path: str,
    model,
    grid_dirs: Dict[int, str],
    no_person_dir: str,
    output_dir: str,
) -> Tuple[str, int]:
    """画像の振り分け処理

    Args:
        image_path: 画像ファイルのパス
        model: YOLOモデル
        grid_dirs: グリッドディレクトリの辞書
        no_person_dir: 人が検出されなかった画像の保存先
        output_dir: 検出結果画像の保存先

    Returns:
        Tuple[str, int]: (grid_info, grid_index_or_none)
            grid_info: 振り分け先の情報文字列
            grid_index_or_none: グリッドインデックス（人が検出されなかった場合は-1）
    """
    print(f"処理中: {image_path}")

    # 人検出とグリッド位置の取得
    detection_result = detect_person_and_get_grid(image_path, model)

    # 検出結果を画像に描画するため、推論も実行
    results = model(image_path)
    result = results[0]
    result_image = result.plot()

    # 出力ファイルパスを設定
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"detected_{output_filename}")

    # 画像を保存
    cv2.imwrite(output_path, result_image)
    print(f"検出結果を保存しました: {output_path}")

    # 画像を振り分け
    if detection_result:
        # 人が検出された場合
        grid_index = detection_result["grid_index"]
        grid_x = detection_result["grid_x"]
        grid_y = detection_result["grid_y"]
        center_x = detection_result["center_x"]
        center_y = detection_result["center_y"]
        confidence = detection_result["confidence"]
        person_count = detection_result["person_count"]

        dest_dir = grid_dirs[grid_index]
        grid_info = f"grid_{grid_index:02d}"

        print(
            f"人を検出: {person_count}人, 中央座標({center_x:.1f}, {center_y:.1f}) -> グリッド位置({grid_x}, {grid_y}) インデックス{grid_index}, 信頼度{confidence:.2f}"
        )

        result_grid_index = grid_index
    else:
        # 人が検出されなかった場合
        dest_dir = no_person_dir
        grid_info = "no-person"
        result_grid_index = -1

    dest_path = os.path.join(dest_dir, os.path.basename(image_path))

    # 画像を移動
    shutil.move(image_path, dest_path)
    print(f"画像を{grid_info}ディレクトリに移動しました")

    return grid_info, result_grid_index


def save_distribution_results(
    output_dir: str, distribution_count: Dict[int, int], no_person_count: int
) -> str:
    """振り分け結果をファイルに保存

    Args:
        output_dir: 出力ディレクトリ
        distribution_count: グリッド別の画像数
        no_person_count: 人が検出されなかった画像数

    Returns:
        str: 結果ファイルのパス
    """
    result_text_path = os.path.join(output_dir, "distribution_result.txt")

    with open(result_text_path, "w", encoding="utf-8") as f:
        f.write("画像振り分け結果\n")
        f.write("=" * 30 + "\n\n")

        # グリッド別の振り分け結果
        f.write("グリッド別振り分け結果:\n")
        total_person_images = 0
        for i in range(16):
            count = distribution_count[i]
            if count > 0:
                grid_x = i % 4
                grid_y = i // 4
                f.write(f"grid_{i:02d} (位置: {grid_x}, {grid_y}): {count}枚\n")
                total_person_images += count

        f.write(f"\nno-person: {no_person_count}枚\n")
        f.write(f"\n合計: {total_person_images + no_person_count}枚\n")
        f.write(f"人検出画像: {total_person_images}枚\n")
        f.write(f"人未検出画像: {no_person_count}枚\n")

    return result_text_path


def print_distribution_results(
    distribution_count: Dict[int, int], no_person_count: int, result_text_path: str
) -> None:
    """振り分け結果をコンソールに表示

    Args:
        distribution_count: グリッド別の画像数
        no_person_count: 人が検出されなかった画像数
        result_text_path: 結果ファイルのパス
    """
    print("\n" + "=" * 50)
    print("振り分け結果:")
    print("=" * 50)
    for i in range(16):
        count = distribution_count[i]
        if count > 0:
            grid_x = i % 4
            grid_y = i // 4
            print(f"grid_{i:02d} (位置: {grid_x}, {grid_y}): {count}枚")

    print(f"no-person: {no_person_count}枚")
    print(f"\n結果をファイルに保存しました: {result_text_path}")


def get_image_files(input_dir: str) -> List[str]:
    """画像ファイルのリストを取得

    Args:
        input_dir: 入力ディレクトリ

    Returns:
        List[str]: 画像ファイルのパスのリスト
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    image_files = []

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(
            file.lower().endswith(ext) for ext in image_extensions
        ):
            image_files.append(file_path)

    return image_files


def main():
    if not os.path.exists("models/yolo11n.engine"):
        # Load a YOLO11n PyTorch model
        model = YOLO("models/yolo11n.pt")

        # Export the model to TensorRT
        model.export(format="engine")  # creates 'yolo11n.engine'

    # Load the exported TensorRT model
    trt_model = YOLO("models/yolo11n.engine", task="detect")

    # 入力ディレクトリ
    input_dir = "images"

    # ディレクトリの設定
    grid_dirs, no_person_dir, output_dir = setup_directories(input_dir)

    # 画像ファイルのリストを取得
    image_files = get_image_files(input_dir)
    print(f"{len(image_files)}枚の画像が見つかりました")

    # 振り分けカウンター
    distribution_count = {i: 0 for i in range(16)}
    no_person_count = 0

    # 各画像に対して推論を実行
    for image_path in image_files:
        grid_info, grid_index = distribute_image(
            image_path, trt_model, grid_dirs, no_person_dir, output_dir
        )

        # カウンターを更新
        if grid_index == -1:
            no_person_count += 1
        else:
            distribution_count[grid_index] += 1

    # 結果の保存と表示
    result_text_path = save_distribution_results(
        output_dir, distribution_count, no_person_count
    )
    print_distribution_results(distribution_count, no_person_count, result_text_path)


if __name__ == "__main__":
    main()
