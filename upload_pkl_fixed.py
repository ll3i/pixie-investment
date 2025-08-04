#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PKL 파일 데이터를 Supabase에 업로드하는 스크립트 (수정본)
- stock_vectors 테이블 생성
- 벡터 데이터를 리스트로 변환하여 업로드
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime
import time

# Python path 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Supabase 클라이언트
from supabase import create_client

class PKLUploader:
    def __init__(self):
        self.SUPABASE_URL = os.getenv('SUPABASE_URL')
        self.SUPABASE_KEY = os.getenv('SUPABASE_KEY')
        
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError("Supabase 환경변수가 설정되지 않았습니다!")
        
        self.supabase = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)
        print("[OK] Supabase 연결 성공!")
        
        self.success_count = 0
        self.error_count = 0
        
        # 파일 경로
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data', 'processed')
        self.web_data_dir = os.path.join(self.base_dir, 'web', 'data')
        print(f"[OK] 데이터 경로: {self.data_dir}")
    
    def create_tables(self):
        """필요한 테이블 생성"""
        print("\n테이블 생성 중...")
        
        # stock_vectors 테이블 생성 SQL
        create_stock_vectors_sql = """
        CREATE TABLE IF NOT EXISTS stock_vectors (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(6) NOT NULL,
            name VARCHAR(100) NOT NULL,
            text TEXT,
            vector JSONB,
            current_price NUMERIC,
            market_cap BIGINT,
            revenue_growth NUMERIC,
            profit_margin NUMERIC,
            debt_ratio NUMERIC,
            per NUMERIC,
            pbr NUMERIC,
            evaluation_score INTEGER,
            evaluation_result VARCHAR(20),
            evaluation_reasons TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker)
        );
        """
        
        # idf_weights 테이블 생성 SQL
        create_idf_weights_sql = """
        CREATE TABLE IF NOT EXISTS idf_weights (
            id SERIAL PRIMARY KEY,
            term VARCHAR(255) NOT NULL UNIQUE,
            idf_value NUMERIC NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            # RPC 호출로 테이블 생성
            self.supabase.rpc('exec_sql', {'query': create_stock_vectors_sql}).execute()
            print("  [OK] stock_vectors 테이블 생성/확인 완료")
        except:
            print("  [INFO] stock_vectors 테이블이 이미 존재하거나 RPC가 없습니다")
        
        try:
            self.supabase.rpc('exec_sql', {'query': create_idf_weights_sql}).execute()
            print("  [OK] idf_weights 테이블 생성/확인 완료")
        except:
            print("  [INFO] idf_weights 테이블이 이미 존재하거나 RPC가 없습니다")
    
    def clear_table(self, table_name):
        """테이블 데이터 삭제"""
        try:
            print(f"  {table_name} 테이블 데이터 삭제 중...")
            # 모든 데이터 삭제
            self.supabase.table(table_name).delete().gte('id', 0).execute()
            print(f"  [OK] {table_name} 테이블 데이터 삭제 완료")
            time.sleep(1)
        except Exception as e:
            print(f"  [WARNING] {table_name} 테이블 삭제 실패 (계속 진행): {str(e)[:100]}")
    
    def upload_batch(self, table_name, records, batch_size=50):
        """배치 단위로 데이터 업로드"""
        total_batches = (len(records) // batch_size) + (1 if len(records) % batch_size else 0)
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                print(f"  배치 {batch_num}/{total_batches} 업로드 중... ", end='')
                self.supabase.table(table_name).insert(batch).execute()
                self.success_count += len(batch)
                print("[OK]")
                time.sleep(0.5)
            except Exception as e:
                self.error_count += len(batch)
                print(f"[ERROR]: {str(e)[:100]}")
                # 첫 번째 레코드의 구조 확인
                if batch:
                    print(f"    샘플 레코드: {json.dumps(batch[0], ensure_ascii=False)[:200]}...")
    
    def upload_stock_vectors(self):
        """stock_data.pkl의 벡터 데이터를 업로드"""
        print("\n1. 주식 벡터 데이터 업로드")
        
        pkl_file = os.path.join(self.data_dir, 'stock_data.pkl')
        if not os.path.exists(pkl_file):
            print(f"  파일을 찾을 수 없습니다: {pkl_file}")
            return
        
        print(f"  파일: stock_data.pkl")
        print(f"  크기: {os.path.getsize(pkl_file) / 1024 / 1024:.2f} MB")
        
        # PKL 파일 로드
        with open(pkl_file, 'rb') as f:
            df = pickle.load(f)
        
        print(f"  전체 {len(df):,}개 레코드")
        
        # 평가 결과 파일도 로드
        eval_file = os.path.join(self.web_data_dir, 'stock_evaluation_results.csv')
        if os.path.exists(eval_file):
            eval_df = pd.read_csv(eval_file, dtype={'종목코드': str}, encoding='utf-8-sig')
            eval_df['종목코드'] = eval_df['종목코드'].astype(str).str.zfill(6)
            
            # 평가 데이터 병합
            df = df.merge(
                eval_df[['종목코드', '평가점수', '종합평가', '평가이유']], 
                on='종목코드', 
                how='left'
            )
            print("  [OK] 평가 데이터 병합 완료")
        
        # 테이블 비우기
        self.clear_table('stock_vectors')
        
        # 레코드 준비
        records = []
        for idx, row in df.iterrows():
            # 벡터를 리스트로 변환
            vector_data = row.get('vector', [])
            if hasattr(vector_data, 'tolist'):
                vector_list = vector_data.tolist()
            elif isinstance(vector_data, list):
                vector_list = vector_data
            else:
                vector_list = []
            
            record = {
                'ticker': str(row['종목코드']),
                'name': str(row['종목명']),
                'text': str(row.get('text', '')),
                'vector': vector_list,  # JSONB로 저장될 리스트
                'current_price': float(row.get('현재가', 0)) if pd.notna(row.get('현재가')) else None,
                'market_cap': int(row.get('시가총액', 0)) if pd.notna(row.get('시가총액')) else None,
                'revenue_growth': float(row.get('매출성장률', 0)) if pd.notna(row.get('매출성장률')) else None,
                'profit_margin': float(row.get('순이익률', 0)) if pd.notna(row.get('순이익률')) else None,
                'debt_ratio': float(row.get('부채비율', 0)) if pd.notna(row.get('부채비율')) else None,
                'per': float(row.get('PER', 0)) if pd.notna(row.get('PER')) else None,
                'pbr': float(row.get('PBR', 0)) if pd.notna(row.get('PBR')) else None,
                'evaluation_score': int(row.get('평가점수', 0)) if pd.notna(row.get('평가점수')) else None,
                'evaluation_result': str(row.get('종합평가', '')) if pd.notna(row.get('종합평가')) else None,
                'evaluation_reasons': str(row.get('평가이유', '')) if pd.notna(row.get('평가이유')) else None
            }
            records.append(record)
        
        # 업로드
        self.upload_batch('stock_vectors', records)
    
    def upload_idf_weights(self):
        """IDF 가중치 데이터 업로드"""
        print("\n2. IDF 가중치 업로드")
        
        pkl_file = os.path.join(self.data_dir, 'idf.pkl')
        if not os.path.exists(pkl_file):
            print(f"  파일을 찾을 수 없습니다: {pkl_file}")
            return
        
        print(f"  파일: idf.pkl")
        print(f"  크기: {os.path.getsize(pkl_file) / 1024 / 1024:.2f} MB")
        
        # PKL 파일 로드
        with open(pkl_file, 'rb') as f:
            idf_data = pickle.load(f)
        
        # 테이블 비우기
        self.clear_table('idf_weights')
        
        # IDF 데이터가 딕셔너리인 경우
        if isinstance(idf_data, dict):
            records = []
            for term, idf_value in idf_data.items():
                record = {
                    'term': str(term)[:255],  # VARCHAR(255) 제한
                    'idf_value': float(idf_value)
                }
                records.append(record)
            
            print(f"  전체 {len(records):,}개 용어")
            self.upload_batch('idf_weights', records, batch_size=100)
        else:
            print("  [WARNING] IDF 데이터가 예상한 형식(dict)이 아닙니다")
    
    def print_summary(self):
        """업로드 요약 출력"""
        print("\n" + "="*60)
        print("업로드 완료!")
        print(f"성공: {self.success_count:,}")
        print(f"실패: {self.error_count:,}")
        print("="*60)

def main():
    print("="*60)
    print("PKL 데이터 Supabase 업로드")
    print("="*60)
    
    try:
        uploader = PKLUploader()
        
        # 테이블 생성
        uploader.create_tables()
        
        # 순서대로 업로드
        uploader.upload_stock_vectors()  # 1. 주식 벡터 데이터
        uploader.upload_idf_weights()    # 2. IDF 가중치
        
        uploader.print_summary()
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("\n환경 변수 확인:")
        print(".env 파일에 SUPABASE_URL이 설정되어 있는지 확인하세요")
        print(".env 파일에 SUPABASE_KEY가 설정되어 있는지 확인하세요")

if __name__ == "__main__":
    main()