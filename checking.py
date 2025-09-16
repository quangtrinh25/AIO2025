import subprocess
import sys
import pkg_resources

def install_packages():
    # Danh sách các gói cần cài đặt (loại bỏ trùng lặp, ưu tiên phiên bản mới nhất)
    packages = [
        'import-ipynb',
        'ipywidgets',
        'Jinja2',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',  # Sửa từ scikit_learn (tên gói chính thức là scikit-learn)
        'torch',
        'tqdm',
        'pandas',  # Trùng lặp, sẽ được xử lý
        'numpy',   # Trùng lặp, sẽ được xử lý
        'xgboost',
        'tqdm',    # Trùng lặp, sẽ được xử lý
        'scikit-learn',  # Trùng lặp, sẽ được xử lý
        'ipywidgets',    # Trùng lặp, sẽ được xử lý
        'imbalanced-learn',
        'torch',   # Trùng lặp, sẽ được xử lý
        'torchvision',
        'torchaudio',
        'nltk',
        'requests',
        'import-ipynb',  # Trùng lặp, sẽ được xử lý
        'captum',
        'transformers',
        'PyRuSH',
        'quicksectx',
        'medspacy',
        'pydicom'
    ]

    # Loại bỏ trùng lặp, giữ nguyên thứ tự xuất hiện đầu tiên
    unique_packages = list(dict.fromkeys(packages))

    # Kiểm tra và cài đặt từng gói
    for package in unique_packages:
        try:
            # Kiểm tra xem gói đã được cài chưa
            pkg = pkg_resources.get_distribution(package)
            print(f"{package} {pkg.version} đã được cài đặt.")
        except pkg_resources.DistributionNotFound:
            print(f"Cài đặt {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Đã cài đặt {package} thành công.")
            except subprocess.CalledProcessError as e:
                print(f"Lỗi khi cài đặt {package}: {e}")
                print("Vui lòng kiểm tra phiên bản Python, tải công cụ bổ sung (như Microsoft C++ Build Tools) nếu cần, hoặc thử cài phiên bản cụ thể.")

if __name__ == "__main__":
    print("Bắt đầu cài đặt các gói...")
    install_packages()
    print("Hoàn tất quá trình cài đặt!")