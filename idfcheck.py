import pickle
import pandas as pd # stock_data.pkl이 DataFrame일 가능성이 높아 함께 import합니다.

# 1. 파일 경로를 변수에 저장합니다.
# 주의: 경로에 있는 역슬래시(\)가 문제를 일으키지 않도록 문자열 앞에 r을 붙여줍니다.
idf_file_path = r"C:\Users\work4\OneDrive\바탕 화면\투자챗봇\data\processed\idf.pkl"


# 2. 'idf.pkl' 파일 열기
try:
    with open(idf_file_path, 'rb') as f:
        idf_data = pickle.load(f)
    
    print("✅ 'idf.pkl' 파일 로드 성공")
    print("---------------------------------")
    print("데이터 타입:", type(idf_data))
    # 데이터의 일부만 확인하고 싶을 때 (데이터가 길 경우 유용)
    # print("데이터 확인:", idf_data[:10]) 
    print("전체 데이터:", idf_data[:])
    print("\n")

except FileNotFoundError:
    print(f"❌ ERROR: '{idf_file_path}' 경로에 파일이 없습니다.")
except Exception as e:
    print(f"❌ 'idf.pkl' 파일 로드 중 오류 발생: {e}")