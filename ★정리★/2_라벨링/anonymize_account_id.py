import pandas as pd
from pathlib import Path

# =========================================
# 1. 폴더 설정
# =========================================

BASE_DIR = Path(__file__).resolve().parent
output_dir = BASE_DIR / "data_anonymized"
output_dir.mkdir(exist_ok=True)

account_map = {}
counter = 1


def get_anonymous_id(value):
    global counter

    if pd.isna(value):
        value = "unknown"
    else:
        value = str(value).strip()

    if value == "":
        value = "unknown"

    if value not in account_map:
        account_map[value] = f"reviewer_{counter:06d}"
        counter += 1

    return account_map[value]


# =========================================
# 2. 현재 폴더의 CSV/XLSX 파일 찾기
# =========================================

files = (
    list(BASE_DIR.glob("all_크롤링.csv"))
)

# 익명화 결과 파일이나 매핑 파일은 제외
files = [
    file for file in files
    if "_anonymized" not in file.stem
    and "mapping" not in file.stem.lower()
]

if not files:
    print("현재 폴더에 처리할 CSV/XLSX 파일이 없습니다.")
    raise SystemExit

print("처리할 파일 목록:")
for file in files:
    print("-", file.name)


# =========================================
# 3. 파일별 account_id 익명화
# =========================================

for file in files:
    print(f"\n처리 중: {file.name}")

    if file.suffix.lower() == ".csv":
        df = pd.read_csv(file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(file)

    if "account_id" not in df.columns:
        print("account_id 컬럼이 없어 건너뜀")
        continue

    df["account_id"] = df["account_id"].apply(get_anonymous_id)

    output_file = output_dir / f"{file.stem}_anonymized.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {output_file}")


# =========================================
# 4. 매핑표 저장
# =========================================
# 이 파일은 원본 account_id와 익명 ID 대응표이므로 GitHub에 올리면 안 됨.

mapping_df = pd.DataFrame(
    [
        {"original_account_id": original, "anonymous_account_id": anon}
        for original, anon in account_map.items()
    ]
)

mapping_file = BASE_DIR / "account_id_mapping_PRIVATE_DO_NOT_UPLOAD.csv"
mapping_df.to_csv(mapping_file, index=False, encoding="utf-8-sig")

print("\n전체 익명화 완료")
print(f"총 익명 계정 수: {len(account_map)}")
print(f"익명화 파일 위치: {output_dir}")
print(f"매핑표 저장 위치: {mapping_file}")
print("주의: account_id_mapping_PRIVATE_DO_NOT_UPLOAD.csv 파일은 GitHub에 올리지 마세요.")