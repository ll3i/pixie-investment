#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV 파일의 컬럼명을 표준화하는 스크립트
"""

import os
import pandas as pd
import glob
from datetime import datetime

def fix_korean_price_columns():
    """한국 주식 가격 CSV 파일의 컬럼명을 표준화"""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    
    # 한국 주식 가격 파일들 찾기
    price_files = glob.glob(os.path.join(data_dir, "kor_price_*.csv"))
    
    print(f"찾은 가격 파일: {len(price_files)}개")
    
    for file_path in price_files:
        try:
            # 파일 읽기
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            print(f"\n파일: {os.path.basename(file_path)}")
            print(f"현재 컬럼: {list(df.columns)}")
            
            # 컬럼명 표준화
            column_mapping = {
                '종목코드': 'ticker',
                '종목명': 'name',
                '날짜': 'date',
                '시가': 'open',
                '고가': 'high',
                '저가': 'low',
                '종가': 'close',
                '거래량': 'volume'
            }
            
            # 컬럼명 변경이 필요한지 확인
            needs_update = False
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    needs_update = True
                    break
            
            if needs_update:
                # 백업 생성
                backup_path = file_path.replace('.csv', '_backup.csv')
                df.to_csv(backup_path, index=False, encoding='utf-8-sig')
                print(f"백업 생성: {os.path.basename(backup_path)}")
                
                # 컬럼명 변경
                df = df.rename(columns=column_mapping)
                
                # 종목코드 6자리로 맞추기
                if 'ticker' in df.columns:
                    df['ticker'] = df['ticker'].astype(str).str.zfill(6)
                
                # 저장
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f"컬럼명 표준화 완료: {list(df.columns)}")
            else:
                print("이미 표준화된 파일")
                
        except Exception as e:
            print(f"오류 발생: {e}")

def check_columns():
    """현재 CSV 파일들의 컬럼 상태 확인"""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    
    # 모든 CSV 파일 패턴
    patterns = [
        "kor_price_*.csv",
        "kor_ticker_*.csv",
        "kor_fs_*.csv",
        "kor_value_*.csv",
        "us_price_*.csv",
        "us_ticker_*.csv"
    ]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(data_dir, pattern))
        if files:
            print(f"\n{pattern}:")
            for file_path in files[:1]:  # 각 패턴별로 첫 번째 파일만
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig', nrows=1)
                    print(f"  파일: {os.path.basename(file_path)}")
                    print(f"  컬럼: {list(df.columns)}")
                except Exception as e:
                    print(f"  오류: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("CSV 컬럼명 표준화 스크립트")
    print("=" * 60)
    
    # 현재 상태 확인
    print("\n1. 현재 컬럼 상태 확인:")
    check_columns()
    
    # 표준화 실행
    print("\n2. 한국 주식 가격 파일 표준화:")
    response = input("\n표준화를 실행하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        fix_korean_price_columns()
        
        # 결과 확인
        print("\n3. 표준화 후 상태:")
        check_columns()
    else:
        print("취소되었습니다.")