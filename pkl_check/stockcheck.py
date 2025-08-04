import pickle
import pandas as pd # stock_data.pkl이 DataFrame일 가능성이 높아 함께 import합니다.

pd.set_option('display.max_columns', None)
# 1. 파일 경로를 변수에 저장합니다.
# 주의: 경로에 있는 역슬래시(\)가 문제를 일으키지 않도록 문자열 앞에 r을 붙여줍니다.

stock_data_file_path = r"C:\Users\work4\OneDrive\바탕 화면\투자챗봇\data\processed\stock_data.pkl"

# 3. 'stock_data.pkl' 파일 열기
try:
    with open(stock_data_file_path, 'rb') as f:
        stock_data = pickle.load(f)

    print("✅ 'stock_data.pkl' 파일 로드 성공")
    print("---------------------------------")
    print("데이터 타입:", type(stock_data))

    # 데이터가 pandas DataFrame인 경우, .head()로 상위 5개 행을 보는 것이 편리합니다.
    if isinstance(stock_data, pd.DataFrame):
        print("데이터 확인 (상위 5개):")
        print(stock_data.head(100))
    else:
        print("전체 데이터:", stock_data)

except FileNotFoundError:
    print(f"❌ ERROR: '{stock_data_file_path}' 경로에 파일이 없습니다.")
except Exception as e:
    print(f"❌ 'stock_data.pkl' 파일 로드 중 오류 발생: {e}")