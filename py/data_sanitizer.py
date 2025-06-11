import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse

# Pillow 的图像大小限制，以防止解压缩炸弹
Image.MAX_IMAGE_PIXELS = None 

# 定义一个合理的尺寸阈值，超过这个尺寸的图片将被等比缩放
# 2048*2048 = 4.1M 像素，对于大多数预训练模型来说足够了
MAX_RESOLUTION_BEFORE_RESIZE = (2048, 2048)
TARGET_SIZE_AFTER_RESIZE = (1024, 1024)

def sanitize_image(args):
    """
    工作函数：验证、缩放和转换单个图像。
    """
    src_path, dest_path = args
    try:
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)

        with Image.open(src_path) as img:
            # 1. 统一转换为 RGB
            img = img.convert("RGB")

            # 2. 检查分辨率并按需缩放
            if img.width > MAX_RESOLUTION_BEFORE_RESIZE[0] or img.height > MAX_RESOLUTION_BEFORE_RESIZE[1]:
                # 按比例缩放，保持长宽比
                img.thumbnail(TARGET_SIZE_AFTER_RESIZE, Image.LANCZOS)

            # 3. 统一保存为高质量 JPEG
            img.save(dest_path, "JPEG", quality=95)
        
        return None  # 表示成功
    except Exception as e:
        return f"Error processing {src_path}: {e}"

def sanitize_dataset(source_dir: str, dest_dir: str, force_rescan=False, workers=None):
    """
    清理和预处理整个数据集。

    Args:
        source_dir (str): 原始数据集的路径 (例如 './raw_data')。
        dest_dir (str): 清理后数据集的存放路径 (例如 './dataset_sanitized')。
        force_rescan (bool): 是否强制重新扫描和处理所有图片。
        workers (int): 使用的工作进程数。
    """
    source_path = Path(source_dir).resolve()
    dest_path = Path(dest_dir).resolve()

    print("--- 启动数据集健康检查与清理程序 ---")
    print(f"原始数据源: {source_path}")
    print(f"目标文件夹: {dest_path}")

    if dest_path.exists() and not force_rescan:
        print("✅ 清理过的目标文件夹已存在，跳过处理。如需强制重新处理，请使用 --force-rescan 标志。")
        return True

    # 搜集所有待处理的图片任务
    tasks = []
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    print("正在搜集图片文件...")
    for ext in supported_formats:
        tasks.extend(source_path.rglob(f"*{ext.lower()}"))
        tasks.extend(source_path.rglob(f"*{ext.upper()}"))

    if not tasks:
        print(f"❌ 在 '{source_path}' 中未找到任何支持的图片。")
        return False
        
    print(f"共找到 {len(tasks)} 张图片需要处理。")

    # 创建处理任务列表
    process_tasks = []
    for src_img_path in tasks:
        relative_p = src_img_path.relative_to(source_path)
        dest_img_path = (dest_path / relative_p).with_suffix(".jpg")
        process_tasks.append((str(src_img_path), str(dest_img_path)))

    # 使用多进程进行处理
    num_workers = workers if workers is not None else min(16, cpu_count())
    errors = []
    print(f"使用 {num_workers} 个工作进程开始处理...")
    with Pool(num_workers) as pool:
        with tqdm(total=len(process_tasks), desc="清理图片中") as pbar:
            for result in pool.imap_unordered(sanitize_image, process_tasks):
                if result:
                    errors.append(result)
                pbar.update()

    if errors:
        print("\n--- ⚠️ 清理过程中发生错误 ---")
        for error in errors[:10]: # 最多显示10条错误
            print(error)
        return False
    
    print("\n✅ 数据集清理成功！")
    return True

if __name__ == '__main__':
    # 允许此脚本被直接调用，以进行手动数据清理
    parser = argparse.ArgumentParser(description="一站式数据集健康检查与预处理工具。")
    parser.add_argument("source", type=str, help="原始数据集的根目录。")
    parser.add_argument("destination", type=str, help="处理后数据的存放目录。")
    parser.add_argument("--workers", type=int, default=None, help="工作进程数（默认：min(16, cpu_cores)）。")
    parser.add_argument("--force-rescan", action="store_true", help="即使目标目录已存在，也强制重新扫描和处理。")
    args = parser.parse_args()
    
    sanitize_dataset(args.source, args.destination, args.force_rescan, args.workers) 
