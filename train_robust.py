import argparse
import sys
import subprocess
from pathlib import Path

# --- 核心修复一：应用猴子补丁 ---
# 在导入任何 ultralytics 内容之前，最先执行此操作
try:
    from py.custom_dataset import CustomClassificationDataset
    import ultralytics.data.dataset
    ultralytics.data.dataset.ClassificationDataset = CustomClassificationDataset
    print("✅ 核心修复: 图片加载器已替换为更稳定的 Pillow 版本。")
except ImportError:
    print("❌ 严重错误: 无法导入 py/custom_dataset.py。请确保该文件存在。")
    sys.exit(1)

# --- 核心修复二：导入数据清理模块 ---
try:
    from py.data_sanitizer import sanitize_dataset
    print("✅ 核心修复: 数据健康检查模块已加载。")
except ImportError:
    print("❌ 严重错误: 无法导入 py/data_sanitizer.py。请确保该文件存在。")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="健壮的 YOLOv8 训练启动器 (一站式解决方案)")
    
    # --- 用户主要配置 ---
    parser.add_argument("--data", type=str, required=True, help="【必需】原始数据集的根目录。")
    parser.add_argument("--model", type=str, default="yolo11x-cls.pt", help="预训练模型的路径或名称。")
    parser.add_argument("--epochs", type=int, default=100, help="训练的总轮数。")
    parser.add_argument("--workers", type=int, default=48, help="数据加载使用的工作进程数。")
    
    # --- 自动化与高级配置 ---
    parser.add_argument("--sanitized-dir", type=str, default="./dataset_sanitized", help="存放清理后数据的目录。")
    parser.add_argument("--force-rescan", action="store_true", help="强制重新扫描和清理数据集，即使目标目录已存在。")
    parser.add_argument("--run-name", type=str, default="robust_run", help="为本次训练运行指定一个清晰的名称。")

    args = parser.parse_args()

    # --- 步骤 1: 自动执行数据健康检查和清理 ---
    success = sanitize_dataset(
        source_dir=args.data,
        dest_dir=args.sanitized_dir,
        force_rescan=args.force_rescan,
        workers=args.workers
    )
    if not success:
        print("❌ 数据集清理失败，训练中止。请检查上面的错误信息。")
        sys.exit(1)

    # --- 步骤 2: 构建并执行最终的 YOLO 训练命令 ---
    print("\n--- 所有检查和预处理均已完成，准备启动训练 ---")
    
    # 我们使用绝对路径以获得最佳兼容性
    sanitized_data_path = str(Path(args.sanitized_dir).resolve())

    command = [
        "yolo",
        "classify",
        f"model={args.model}",
        f"data={sanitized_data_path}",
        f"epochs={args.epochs}",
        f"workers={args.workers}",
        f"name={args.run_name}",
        "cache=false" # 建议禁用缓存，因为我们的加载器很快，可以避免缓存问题
    ]
    
    print(f"执行命令: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
        print("\n🎉 训练成功完成！")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\n❌ 训练执行失败: {e}")
        print("请确保 'yolo' 命令在您的 Conda 环境中可用且路径正确。")
        sys.exit(1)

if __name__ == "__main__":
    main() 