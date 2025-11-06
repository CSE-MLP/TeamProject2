import os
import zipfile
import kaggle

# 환경변수에서 Kaggle 인증 정보 가져오기
username = os.getenv("KAGGLE_USERNAME")
key = os.getenv("KAGGLE_KEY")

if not username or not key:
    raise EnvironmentError("KAGGLE_USERNAME and KAGGLE_KEY must be set as environment variables.")

# Kaggle API 인증 설정
os.environ['KAGGLE_USERNAME'] = username
os.environ['KAGGLE_KEY'] = key

# 대회명 및 다운로드 경로
competition_name = "llm-classification-finetuning"
download_path = "./data"
os.makedirs(download_path, exist_ok=True)

# 데이터 다운로드
print(f"Downloading data for competition: {competition_name}")
kaggle.api.competition_download_files(competition_name, path=download_path)

# 압축 해제
zip_path = os.path.join(download_path, f"{competition_name}.zip")
if os.path.exists(zip_path):
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    os.remove(zip_path)
    print("Extraction complete!")
else:
    print("No zip file found.")

print(f"Data ready at: {download_path}")

