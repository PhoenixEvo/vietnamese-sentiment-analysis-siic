"""
Main pipeline script for Emotion Detection project
"""

import argparse
import os
import subprocess
import sys


def run_preprocessing():
    """Run data preprocessing"""
    print(" Chạy Data Preprocessing...")
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from siic.data.preprocessors import main; main()"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(" Preprocessing hoàn thành!")
            print(result.stdout)
        else:
            print(" Lỗi trong preprocessing:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f" Lỗi khi chạy preprocessing: {e}")
        return False

    return True


def run_training():
    """Run model training"""
    print("\n Chạy Model Training...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--model", "baselines"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(" Training hoàn thành!")
            print(result.stdout)
        else:
            print(" Lỗi trong training:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f" Lỗi khi chạy training: {e}")
        return False

    return True


def run_dashboard():
    """Run Streamlit dashboard"""
    print("\n Khởi động Dashboard...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
    except Exception as e:
        print(f" Lỗi khi chạy dashboard: {e}")


def install_dependencies():
    """Install required dependencies"""
    print("📦 Cài đặt dependencies...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(" Dependencies đã được cài đặt!")
        else:
            print(" Lỗi khi cài đặt dependencies:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f" Lỗi khi cài đặt dependencies: {e}")
        return False

    return True


def setup_directories():
    """Create necessary directories"""
    print("📁 Tạo các thư mục cần thiết...")

    directories = ["data", "data/raw", "data/processed", "models", "results", "notebooks"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"    {directory}")

    print(" Đã tạo các thư mục!")


def download_uit_vsfc():
    """Download UIT-VSFC dataset"""
    print("\n Downloading UIT-VSFC dataset...")
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from siic.data.loaders import main; main()"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(" UIT-VSFC download hoàn thành!")
            print(result.stdout)
        else:
            print(" Lỗi khi download UIT-VSFC:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f" Lỗi khi download UIT-VSFC: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Emotion Detection Pipeline")
    parser.add_argument(
        "--step",
        choices=["setup", "install", "download", "preprocess", "train", "dashboard", "all"],
        default="all",
        help="Chọn bước thực hiện",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EMOTION DETECTION PROJECT PIPELINE")
    print("=" * 60)
    print("Team: InsideOut")
    print("Leader: Nguyễn Nhật Phát")
    print("Member: Nguyễn Tiến Huy")
    print("=" * 60)

    if args.step in ["setup", "all"]:
        setup_directories()

    if args.step in ["install", "all"]:
        if not install_dependencies():
            print(" Pipeline dừng do lỗi cài đặt dependencies")
            return

    if args.step in ["download", "all"]:
        response = input("\nBạn có muốn download UIT-VSFC dataset thực tế? (y/n, mặc định: n): ")
        if response.lower() in ["y", "yes", "có"]:
            if not download_uit_vsfc():
                print(" Pipeline dừng do lỗi download dataset")
                return
        else:
            print("⏭️ Bỏ qua download, sẽ sử dụng sample data")

    if args.step in ["preprocess", "all"]:
        if not run_preprocessing():
            print(" Pipeline dừng do lỗi preprocessing")
            return

    if args.step in ["train", "all"]:
        if not run_training():
            print(" Pipeline dừng do lỗi training")
            return

    if args.step in ["dashboard", "all"]:
        print("\n Pipeline hoàn thành!")
        print("\nBạn có thể:")
        print("1. Chạy dashboard: streamlit run dashboard/app.py")
        print("2. Xem kết quả trong thư mục results/")
        print("3. Sử dụng trained models trong thư mục models/")

        response = input("\nCó muốn chạy dashboard ngay bây giờ? (y/n): ")
        if response.lower() in ["y", "yes", "có"]:
            run_dashboard()


if __name__ == "__main__":
    main()
