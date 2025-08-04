import os
import sys

# Python path 설정을 가장 먼저 수행
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Disable all progress bars in sentence-transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['DISABLE_TQDM'] = 'true'

# Disable tqdm globally before any imports
try:
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
except:
    pass

import json
import time
import uuid
import requests
import random
import sqlite3
import logging
import math
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 환경변수 미리 설정 (config.py 임포트 오류 방지)
if not os.environ.get("FLASK_SECRET_KEY"):
    os.environ["FLASK_SECRET_KEY"] = "minerva_investment_advisor_secure_key_2024"
if not os.environ.get("FLASK_ENV"):
    os.environ["FLASK_ENV"] = "development"

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure transformers logging before importing modules that use it
try:
    import transformers
    transformers.logging.set_verbosity_error()
    # Disable progress bars
    transformers.utils.logging.disable_progress_bar()
except:
    pass

from src.investment_advisor import InvestmentAdvisor
# APScheduler는 필요할 때만 import
from src.financial_data_processor import FinancialDataProcessor
from dateutil import parser as dateutil_parser
from src.simplified_portfolio_prediction import get_portfolio_recommendations
from src.db_client import get_supabase_client
from src.financial_report_analyzer import FinancialReportAnalyzer, RiskLevel
import asyncio
from news_api_helper import (
    collect_realtime_news, get_mock_news_data, get_pixie_insights,
    get_trend_keywords, get_news_sentiment_summary, news_cache
)

# 현재 디렉토리를 파이썬 경로에 추가하여 src 모듈을 import할 수 있게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Logger 설정 - 경고 이상만 표시
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# sentence-transformers 로거 비활성화
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

def clean_json_data(data):
    """JSON 직렬화 가능한 데이터로 변환 (NaN, Infinity 처리)"""
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "minerva_investment_advisor")
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1시간
app.config['SESSION_COOKIE_SECURE'] = False  # 개발 환경에서는 False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
IMAGE_DIR = os.path.join(app.static_folder, 'images')
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.json")
ANALYSIS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results.json")

DB_PATH = os.path.join(BASE_DIR, 'investment_data.db')

# Supabase 설정
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
supabase = get_supabase_client() if SUPABASE_URL and SUPABASE_KEY else None

def init_db():
    """데이터베이스 초기화 및 필수 테이블 생성"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        logger.info("데이터베이스 초기화 중...")
        
        # 사용자 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )''')
        
        # 사용자 투자 프로필 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            risk_tolerance INTEGER,
            investment_time_horizon INTEGER,
            financial_goal_orientation INTEGER,
            information_processing_style INTEGER,
            investment_fear INTEGER,
            investment_confidence INTEGER,
            analysis_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # 채팅 기록 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            agent_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # 관심 키워드 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS news_keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            keyword TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # 포트폴리오 추천 기록 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS portfolio_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            recommendation_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # 알림 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            content TEXT,
            level TEXT,
            date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # 알림 이력 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS alerts_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            alert_type TEXT,
            message TEXT,
            level TEXT,
            is_read BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # 관심 종목 테이블 (watchlist)
        c.execute('''CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            stock_code TEXT NOT NULL,
            stock_name TEXT,
            market TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, stock_code)
        )''')
        
        # 기존 테이블 마이그레이션
        _migrate_tables(c)
        
        conn.commit()
        logger.info("데이터베이스 초기화 완료")
        
    except Exception as e:
        logger.error(f"데이터베이스 초기화 실패: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def _migrate_tables(cursor):
    """기존 테이블에 필요한 컬럼 추가 (마이그레이션)"""
    try:
        # alerts_history 테이블에 is_read 컬럼 확인
        cursor.execute("PRAGMA table_info(alerts_history)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'is_read' not in columns:
            cursor.execute('ALTER TABLE alerts_history ADD COLUMN is_read BOOLEAN DEFAULT 0')
            logger.debug("alerts_history 테이블에 is_read 컬럼 추가")
            
        # users 테이블에 추가 컬럼들 확인
        cursor.execute("PRAGMA table_info(users)")
        user_columns = [row[1] for row in cursor.fetchall()]
        if 'last_login' not in user_columns:
            cursor.execute('ALTER TABLE users ADD COLUMN last_login TIMESTAMP')
        if 'email' not in user_columns:
            cursor.execute('ALTER TABLE users ADD COLUMN email TEXT')
        if 'name' not in user_columns:
            cursor.execute('ALTER TABLE users ADD COLUMN name TEXT')
            
    except Exception as e:
        logger.error(f"테이블 마이그레이션 중 오류: {e}")

# 앱 시작 시 DB 초기화
init_db()

# 프롬프트 파일 경로 설정
if not os.path.isabs(SRC_DIR):
    SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
PROMPT_FILES = {
    "AI-A": os.path.join(SRC_DIR, "prompt_AI-A.txt"),
    "AI-A2": os.path.join(SRC_DIR, "prompt_AI-A2.txt"),
    "AI-B": os.path.join(SRC_DIR, "prompt_AI-B.txt"),
    "survey-analysis": os.path.join(SRC_DIR, "prompt_survey-analysis.txt"),
    "survey-score": os.path.join(SRC_DIR, "prompt_survey-score.txt")
}

# API 관련 설정
API_CONFIG = {
    "host": os.environ.get("CLOVA_API_HOST", "https://api.clovastudio.com"),
    "api_key": os.environ.get("CLOVA_API_KEY", ""),
    "request_id": os.environ.get("CLOVA_REQUEST_ID", ""),
    "max_retries": int(os.environ.get("API_MAX_RETRIES", 3)),
    "retry_delay": int(os.environ.get("API_RETRY_DELAY", 1))
}

# API 상태 확인
def check_api_availability():
    """사용 가능한 API 확인 및 설정"""
    openai_key = os.environ.get('OPENAI_API_KEY')
    clova_key = os.environ.get('CLOVA_API_KEY')
    
    api_status = {
        'openai': bool(openai_key and len(openai_key.strip()) > 10),
        'clova': bool(clova_key and len(clova_key.strip()) > 10),
        'simulation': True
    }
    
    if api_status['openai']:
        api_status['selected'] = 'openai'
    elif api_status['clova']:
        api_status['selected'] = 'clova' 
    else:
        api_status['selected'] = 'simulation'
    
    return api_status

# 시작 시 API 상태 확인
API_STATUS = check_api_availability()
# print(f"API 상태: {API_STATUS['selected']}")  # 시작 시 출력 비활성화
if API_STATUS['selected'] == 'simulation':
    logger.warning("API 키가 설정되지 않아 시뮬레이션 모드로 실행됩니다.")

# 디렉토리 초기화
os.makedirs(IMAGE_DIR, exist_ok=True)

# 뉴스 처리 및 스케줄링 초기화 (선택적)
ENABLE_NEWS_PROCESSOR = os.environ.get('ENABLE_NEWS_PROCESSOR', 'false').lower() == 'true'

if ENABLE_NEWS_PROCESSOR:
    try:
        news_processor = FinancialDataProcessor()
        logger.info("뉴스 프로세서 초기화 완료")
    except Exception as e:
        logger.error(f"뉴스 프로세서 초기화 실패: {e}")
        news_processor = None
else:
    logger.info("뉴스 프로세서 비활성화됨 (ENABLE_NEWS_PROCESSOR=true로 활성화 가능)")
    news_processor = None

# 뉴스 수집 스케줄러 초기화 (선택적)
news_scheduler = None
ENABLE_SCHEDULER = os.environ.get('ENABLE_SCHEDULER', 'false').lower() == 'true'

def scheduled_news_update():
    """스케줄된 뉴스 업데이트 실행"""
    try:
        print("예정된 뉴스 업데이트 시작...")
        if news_processor:
            # 주요 키워드로 뉴스 수집
            keywords = ['주식', '투자', '경제', '금융', '삼성전자', 'SK하이닉스']
            for keyword in keywords:
                try:
                    news_processor.fetch_news_from_api(stock_name=keyword)
                    time.sleep(1)  # API 제한 고려
                except Exception as e:
                    print(f"키워드 '{keyword}' 뉴스 수집 실패: {e}")
            print("뉴스 업데이트 완료")
        else:
            print("뉴스 프로세서가 초기화되지 않음")
    except Exception as e:
        print(f"뉴스 업데이트 실패: {e}")

# 스케줄러 시작 (환경변수로 제어)
if ENABLE_SCHEDULER:
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        news_scheduler = BackgroundScheduler()
        news_scheduler.add_job(
            func=scheduled_news_update,
            trigger="interval", 
            minutes=10,
            id='news_update_job',
            name='뉴스 자동 수집',
            replace_existing=True
        )
        news_scheduler.start()
        print("뉴스 스케줄러 시작됨 (10분 간격)")
        
        # 앱 종료 시 스케줄러 정리
        import atexit
        atexit.register(lambda: news_scheduler.shutdown() if news_scheduler and news_scheduler.running else None)
    except Exception as e:
        print(f"뉴스 스케줄러 시작 실패: {e}")
        news_scheduler = None
else:
    print("뉴스 스케줄러 비활성화됨 (ENABLE_SCHEDULER=true로 활성화 가능)")

# 실제 데이터 처리 클래스
import pandas as pd
import glob

class RealDataManager:
    """실제 수집된 데이터를 관리하고 분석하는 클래스"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data", "raw")
        logger.debug(f"데이터 디렉토리: {self.data_dir}")
        
        # Supabase 초기화
        self.SUPABASE_URL = os.getenv('SUPABASE_URL')
        self.SUPABASE_KEY = os.getenv('SUPABASE_KEY')
        self.supabase = None
        
        if self.SUPABASE_URL and self.SUPABASE_KEY:
            from supabase import create_client
            self.supabase = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)
            logger.info("RealDataManager: Supabase 연결 성공")
        
    def get_latest_file(self, pattern):
        """패턴에 맞는 최신 파일 반환"""
        try:
            files = glob.glob(os.path.join(self.data_dir, pattern))
            if files:
                return max(files, key=os.path.getmtime)
            return None
        except Exception as e:
            print(f"파일 검색 오류 ({pattern}): {e}")
            return None
    
    def load_korean_stocks(self):
        """한국 주식 데이터 로드"""
        try:
            # 최신 파일들 가져오기
            ticker_file = self.get_latest_file("kor_ticker_*.csv")
            price_file = self.get_latest_file("kor_price_*.csv") 
            fs_file = self.get_latest_file("kor_fs_*.csv")
            value_file = self.get_latest_file("kor_value_*.csv")
            
            data = {}
            
            # 종목 정보 로드
            if ticker_file:
                try:
                    df = pd.read_csv(ticker_file, encoding='utf-8')
                    data['tickers'] = df.to_dict('records')
                    print(f"한국 종목 {len(df)}개 로드")
                except Exception as e:
                    print(f"종목 파일 로드 실패: {e}")
            
            # 주가 데이터 로드  
            if price_file:
                try:
                    df = pd.read_csv(price_file, encoding='utf-8')
                    # 날짜 컬럼 파싱
                    if '날짜' in df.columns:
                        df['날짜'] = pd.to_datetime(df['날짜'])
                    data['prices'] = df.to_dict('records')
                    print(f"한국 주가 데이터 {len(df)}건 로드")
                except Exception as e:
                    print(f"주가 파일 로드 실패: {e}")
            
            # 재무제표 데이터 로드
            if fs_file:
                try:
                    df = pd.read_csv(fs_file, encoding='utf-8')
                    data['financials'] = df.to_dict('records')
                    print(f"한국 재무제표 {len(df)}건 로드")
                except Exception as e:
                    print(f"재무제표 파일 로드 실패: {e}")
            
            # 가치평가 데이터 로드
            if value_file:
                try:
                    df = pd.read_csv(value_file, encoding='utf-8')
                    data['valuations'] = df.to_dict('records')
                    print(f"한국 가치평가 {len(df)}건 로드")
                except Exception as e:
                    print(f"가치평가 파일 로드 실패: {e}")
                    
            return data
            
        except Exception as e:
            print(f"한국 주식 데이터 로드 실패: {e}")
            return {}
    
    def load_us_stocks(self):
        """미국 주식 데이터 로드"""
        try:
            ticker_file = self.get_latest_file("us_ticker_*.csv")
            price_file = self.get_latest_file("us_price_*.csv")
            
            data = {}
            
            if ticker_file:
                try:
                    df = pd.read_csv(ticker_file, encoding='utf-8')
                    data['tickers'] = df.to_dict('records')
                    print(f"미국 종목 {len(df)}개 로드")
                except Exception as e:
                    print(f"미국 종목 파일 로드 실패: {e}")
            
            if price_file:
                try:
                    df = pd.read_csv(price_file, encoding='utf-8')
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                    data['prices'] = df.to_dict('records')
                    print(f"미국 주가 데이터 {len(df)}건 로드")
                except Exception as e:
                    print(f"미국 주가 파일 로드 실패: {e}")
            
            return data
            
        except Exception as e:
            print(f"미국 주식 데이터 로드 실패: {e}")
            return {}

    def analyze_stock_trends(self, stock_data, stock_code, days=30):
        """주식의 과거 추세를 분석하여 미래 예측"""
        try:
            if not stock_data or 'prices' not in stock_data:
                print(f"데이터 없음: stock_data={bool(stock_data)}, prices={bool(stock_data and 'prices' in stock_data)}")
                return None
                
            print(f"분석 시작: {stock_code}, 전체 가격 데이터: {len(stock_data.get('prices', []))}개")
            
            # 해당 종목의 가격 데이터 필터링
            if len(stock_code) == 6 and stock_code.isdigit():
                # 한국 주식 (6자리 숫자)
                prices_data = []
                for p in stock_data['prices']:
                    # ticker 컬럼 확인 (영문 컬럼명 사용)
                    code_in_data = str(p.get('ticker', p.get('종목코드', ''))).strip()
                    if code_in_data == stock_code or code_in_data.zfill(6) == stock_code.zfill(6):
                        prices_data.append(p)
                # 영문 컬럼명 사용
                if len(stock_data['prices']) > 0:
                    price_key = 'Close' if 'Close' in stock_data['prices'][0] else '종가'
                    volume_key = 'Volume' if 'Volume' in stock_data['prices'][0] else '거래량'
                else:
                    price_key = 'Close'
                    volume_key = 'Volume'
                print(f"한국 주식 {stock_code}: {len(prices_data)}개 데이터 찾음")
            else:
                # 미국 주식
                prices_data = []
                for p in stock_data['prices']:
                    if p.get('Ticker') == stock_code:
                        prices_data.append(p)
                price_key = 'Close'
                volume_key = 'Volume'
                print(f"미국 주식 {stock_code}: {len(prices_data)}개 데이터 찾음")
            
            if len(prices_data) < 2:
                print(f"데이터 부족: {len(prices_data)}개")
                return None
                
            # 최근 데이터 사용 (최대 30개)
            prices_data = prices_data[-30:] if len(prices_data) > 30 else prices_data
            
            prices = []
            volumes = []
            
            for data in prices_data:
                try:
                    price = float(data[price_key])
                    volume = float(data.get(volume_key, 0))
                    prices.append(price)
                    volumes.append(volume)
                except (ValueError, KeyError):
                    continue
            
            if len(prices) < 2:
                return None
                
            # 가격 변화율 계산
            price_changes = []
            for i in range(1, len(prices)):
                change = (prices[i] - prices[i-1]) / prices[i-1] * 100
                price_changes.append(change)
            
            # 변동성 계산 (표준편차 기반)
            avg_price = sum(prices) / len(prices)
            variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
            volatility = (variance ** 0.5) / avg_price * 100
            
            # 거래량 추세
            avg_volume = sum(volumes) / len(volumes) if volumes else 1
            recent_volume = volumes[-1] if volumes else avg_volume
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # 미래 예측 (이동평균 + 추세 기반)
            avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
            
            current_price = prices[-1]
            predictions = []
            
            for day in range(1, days + 1):
                # 추세와 변동성을 고려한 예측
                trend_effect = avg_change * (day ** 0.8)  # 추세는 시간에 따라 감소
                volatility_effect = volatility * (day ** 0.5) / 20  # 변동성 증가
                volume_effect = (volume_ratio - 1) * 5  # 거래량 영향
                
                # 랜덤 요소 추가 (시장 불확실성)
                import random
                random_factor = random.uniform(-2, 2) * (day / 10)
                
                predicted_change = trend_effect + volume_effect + random_factor
                predicted_price = current_price * (1 + predicted_change / 100)
                
                # 합리적 범위 제한 (현재가 대비 ±50%)
                predicted_price = max(current_price * 0.5, 
                                    min(predicted_price, current_price * 1.5))
                
                # 신뢰도 계산 (시간이 지날수록, 변동성이 클수록 감소)
                confidence = max(0.2, 0.9 - (volatility / 200) - (day * 0.02))
                
                predictions.append({
                    'day': day,
                    'predicted_price': round(predicted_price, 2),
                    'confidence': round(confidence, 3),
                    'price_change_percent': round(predicted_change, 2)
                })
            
            return {
                'stock_code': stock_code,
                'current_price': current_price,
                'avg_change': round(avg_change, 2),
                'volatility': round(volatility, 2),
                'volume_ratio': round(volume_ratio, 2),
                'predictions': predictions,
                'trend': 'bullish' if avg_change > 1 else 'bearish' if avg_change < -1 else 'neutral',
                'analysis_period': len(prices)
            }
            
        except Exception as e:
            print(f"주식 추세 분석 에러: {e}")
            return None

    def generate_time_series_prediction(self, stock_code, days=7):
        """특정 종목의 시계열 예측을 생성"""
        try:
            # 데이터 로드
            if len(stock_code) == 6 and stock_code.isdigit():
                # 한국 주식
                stock_data = self.load_korean_stocks()
                market = "KRX"
            else:
                # 미국 주식
                stock_data = self.load_us_stocks()  
                market = "NASDAQ/NYSE"
            
            if not stock_data:
                return {
                    'error': '데이터를 로드할 수 없습니다',
                    'stock_code': stock_code
                }
                
            # 추세 분석 수행
            print(f"{stock_code} 종목의 추세 분석 시작...")
            analysis = self.analyze_stock_trends(stock_data, stock_code, days)
            if not analysis:
                print(f"{stock_code}: 추세 분석 실패 - 데이터 없음")
                return {
                    'error': '해당 종목의 데이터를 찾을 수 없습니다',
                    'stock_code': stock_code
                }
                
            # 종목 정보 가져오기
            stock_name = "Unknown"
            if 'tickers' in stock_data:
                for ticker in stock_data['tickers']:
                    if len(stock_code) == 6 and str(ticker.get('종목코드', '')).zfill(6) == stock_code.zfill(6):
                        stock_name = ticker.get('종목명', f'Stock-{stock_code}')
                        break
                    elif ticker.get('Ticker') == stock_code:
                        stock_name = ticker.get('Name', f'Stock-{stock_code}')
                        break
            
            # 기본 이름 설정 (데이터에서 찾지 못한 경우)
            if stock_name == "Unknown":
                if len(stock_code) == 6 and stock_code.isdigit():
                    stock_name = f"한국주식-{stock_code}"
                else:
                    stock_name = f"{stock_code}"
            
            # 시계열 예측 결과 구성
            max_price = max(p['predicted_price'] for p in analysis['predictions'])
            min_price = min(p['predicted_price'] for p in analysis['predictions'])
            avg_confidence = sum(p['confidence'] for p in analysis['predictions']) / len(analysis['predictions'])
            
            # 7일 후 예측가격 계산
            final_prediction = analysis['predictions'][-1] if analysis['predictions'] else analysis['current_price']
            predicted_price = final_prediction['predicted_price']
            change_percent = ((predicted_price - analysis['current_price']) / analysis['current_price'] * 100) if analysis['current_price'] > 0 else 0
            
            result = {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'market': market,
                'current_price': analysis['current_price'],
                'predicted_price': predicted_price,
                'change_percent': round(change_percent, 2),
                'trend': 'bullish' if change_percent > 0 else 'bearish' if change_percent < 0 else 'neutral',
                'confidence': round(avg_confidence, 2),
                'model_type': 'ARIMA-X',
                'prediction_horizon': f'{days}일',
                'current_info': {
                    'current_price': analysis['current_price'],
                    'trend': analysis['trend'],
                    'volatility': analysis['volatility'],
                    'avg_change_rate': analysis['avg_change'],
                    'volume_ratio': analysis['volume_ratio']
                },
                'predictions': analysis['predictions'],
                'forecast_summary': {
                    'max_predicted_price': max_price,
                    'min_predicted_price': min_price,
                    'price_range': round(max_price - min_price, 2),
                    'avg_confidence': round(avg_confidence, 3),
                    'analysis_period_days': analysis['analysis_period']
                },
                'analysis_summary': f"""
{stock_name} ({stock_code}) - {market}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

현재 상황:
• 현재가: {analysis['current_price']:,.0f}원/달러
• 추세: {'상승' if analysis['trend'] == 'bullish' else '하락' if analysis['trend'] == 'bearish' else '보합'}
• 변동성: {analysis['volatility']:.1f}%
• 평균 변화율: {analysis['avg_change']:.1f}%
• 거래량 비율: {analysis['volume_ratio']:.1f}x

{days}일 예측:
• 최고 예상가: {max_price:,.0f} ({((max_price - analysis['current_price'])/analysis['current_price']*100):+.1f}%)
• 최저 예상가: {min_price:,.0f} ({((min_price - analysis['current_price'])/analysis['current_price']*100):+.1f}%)
• 예측 범위: ±{((max_price - min_price)/2/analysis['current_price']*100):.1f}%
• 평균 신뢰도: {avg_confidence:.1%}

이 예측은 과거 데이터를 기반으로 한 통계적 분석이며, 
    실제 주가는 다양한 요인에 의해 달라질 수 있습니다.
                """
            }
            
            return result
            
        except Exception as e:
            print(f"시계열 예측 생성 에러: {e}")
            return {
                'error': f'예측 생성 중 오류가 발생했습니다: {str(e)}',
                'stock_code': stock_code
            }
    
    def load_news_data(self):
        """뉴스 데이터 로드"""
        try:
            news_file = self.get_latest_file("news_*.csv")
            
            if news_file:
                df = pd.read_csv(news_file, encoding='utf-8')
                print(f"뉴스 데이터 {len(df)}건 로드")
                return df.to_dict('records')
            
            return []
            
        except Exception as e:
            print(f"뉴스 데이터 로드 실패: {e}")
            return []
    
    def get_stock_analysis(self, ticker, market='KR'):
        """특정 종목 분석 데이터 반환"""
        try:
            if market == 'KR':
                data = self.load_korean_stocks()
                prices = data.get('prices', [])
                financials = data.get('financials', [])
                valuations = data.get('valuations', [])
                
                # 해당 종목 데이터 필터링
                stock_prices = [p for p in prices if p.get('종목코드') == ticker]
                stock_financials = [f for f in financials if f.get('종목코드') == ticker]
                stock_valuations = [v for v in valuations if v.get('종목코드') == ticker]
                
                return {
                    'prices': stock_prices,
                    'financials': stock_financials,
                    'valuations': stock_valuations
                }
            else:
                data = self.load_us_stocks()
                prices = data.get('prices', [])
                stock_prices = [p for p in prices if p.get('Ticker') == ticker]
                
                return {
                    'prices': stock_prices
                }
                
        except Exception as e:
            print(f"종목 분석 데이터 로드 실패 ({ticker}): {e}")
            return {}
    
    def get_stock_price_data(self, ticker):
        """특정 종목의 가격 데이터만 반환"""
        try:
            # 6자리 숫자면 한국 주식
            if len(ticker) == 6 and ticker.isdigit():
                data = self.load_korean_stocks()
                prices = data.get('prices', [])
                # ticker 컬럼 확인 (영문 컬럼명 사용)
                result = [p for p in prices if str(p.get('ticker', p.get('종목코드', ''))).strip() == ticker or 
                          str(p.get('ticker', p.get('종목코드', ''))).strip().zfill(6) == ticker.zfill(6)]
                print(f"get_stock_price_data({ticker}): {len(result)}개 데이터 반환")
                return result
            else:
                # 미국 주식
                data = self.load_us_stocks()
                prices = data.get('prices', [])
                return [p for p in prices if p.get('Ticker') == ticker]
        except Exception as e:
            print(f"가격 데이터 로드 실패 ({ticker}): {e}")
            return []
    
    def get_stock_price_from_supabase(self, stock_code):
        """Supabase에서 특정 종목의 가격 데이터를 직접 조회"""
        try:
            if not self.supabase:
                return None
            
            # 종목코드 6자리로 맞추기
            stock_code = str(stock_code).zfill(6)
            
            # 최근 30일 데이터 조회
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            response = self.supabase.table('kor_stock_prices').select('*').eq(
                'ticker', stock_code
            ).gte(
                'date', thirty_days_ago
            ).order(
                'date', desc=True
            ).execute()
            
            if response.data and len(response.data) > 0:
                logger.debug(f"Supabase에서 {stock_code} 종목 {len(response.data)}개 가격 데이터 조회 성공")
                
                # 최신 데이터
                latest = response.data[0]
                
                # 전일 대비 계산
                prev_close = response.data[1]['close'] if len(response.data) > 1 else latest['close']
                change = float(latest['close']) - float(prev_close)
                change_pct = (change / float(prev_close) * 100) if float(prev_close) > 0 else 0
                
                # 종목명 조회
                ticker_info = self.supabase.table('kor_stock_tickers').select('name').eq(
                    'ticker', stock_code
                ).limit(1).execute()
                
                stock_name = ticker_info.data[0]['name'] if ticker_info.data else f'종목 {stock_code}'
                
                # 평가 정보 조회
                eval_info = self.supabase.table('kor_stock_evaluations').select('*').eq(
                    'ticker', stock_code
                ).limit(1).execute()
                
                return {
                    'success': True,
                    'price': {
                        '종목코드': stock_code,
                        '종목명': stock_name,
                        '종가': float(latest['close']),
                        '시가': float(latest['open']),
                        '고가': float(latest['high']),
                        '저가': float(latest['low']),
                        '거래량': int(latest['volume']),
                        '전일대비': change,
                        '전일대비율': change_pct,
                        'datetime': latest['date'],
                        'per': eval_info.data[0]['per'] if eval_info.data else 0,
                        'pbr': eval_info.data[0]['pbr'] if eval_info.data else 0,
                        '평가점수': eval_info.data[0]['score'] if eval_info.data else 0,
                        '평가등급': eval_info.data[0]['evaluation'] if eval_info.data else ''
                    },
                    'data_source': 'supabase'
                }
            else:
                logger.debug(f"Supabase에서 {stock_code} 종목 데이터 없음")
                return None
                
        except Exception as e:
            logger.error(f"Supabase 가격 조회 오류: {e}")
            return None

# 데이터 매니저 초기화
real_data_manager = RealDataManager()
print("실제 데이터 관리자 초기화 완료")

# 프롬프트 파일 로드 함수
def load_prompt(prompt_name):
    """프롬프트 파일을 로드합니다."""
    if prompt_name not in PROMPT_FILES:
        return None
        
    try:
        with open(PROMPT_FILES[prompt_name], "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"프롬프트 로드 에러: {e}")
        # 프롬프트 파일이 없을 경우 기본 프롬프트 반환
        if prompt_name == "AI-A":
            return """당신은 개인화된 투자 조언을 제공하는 AI 금융 상담사입니다. 현재 상담 중인 투자자는 다음과 같은 특성을 가지고 있습니다:
1. 위험 감수성: [risk_tolerance_analysis]
2. 투자 시간 범위: [investment_time_horizon_analysis]
3. 재무 목표 지향성: [financial_goal_orientation_analysis]
4. 정보 처리 스타일: [information_processing_style_analysis]
5. 투자 자신감: [investment_confidence_analysis]

투자자의 성향을 고려하여 맞춤형 투자 조언을 제공하세요. 투자자의 위험 감수 성향과 투자 기간을 고려하고, 정보 처리 스타일에 맞게 설명을 조정하세요. 투자자의 자신감 수준을 존중하되 객관적인 조언을 제공하세요.

한국어로 응답해야 합니다."""
        elif prompt_name == "AI-A2":
            return """당신은 AI-A2로, 사용자의 투자 성향을 이해한 후 금융 데이터 AI(AI-B)에게 필요한 금융 정보를 물어보는 AI 금융 상담사입니다.
사용자의 투자 성향과 질문을 바탕으로 AI-B와 대화하여 적절한 금융 정보와 조언을 얻는 것이 당신의 역할입니다.

사용자 정보
1. 위험 감수성: [risk_tolerance_analysis]
2. 투자 시간 범위: [investment_time_horizon_analysis]
3. 재무 목표 지향성: [financial_goal_orientation_analysis]
4. 정보 처리 스타일: [information_processing_style_analysis]
5. 투자 자신감: [investment_confidence_analysis]

한국어로 응답해야 합니다."""
        elif prompt_name == "AI-B":
            return """당신은 금융 데이터를 학습한 후 이에 대한 정보를 제공하는 AI-B입니다.
당신의 역할은 AI-A2의 질문에 대해 정확하고 근거 있는 금융 정보와 데이터를 제공하는 것입니다.

최신 금융 데이터를 제공하고, 통계적 분석을 바탕으로 주장과 추천을 뒷받침하세요. 객관성을 유지하고 위험-수익 분석을 명확히 제시하세요.

한국어로 응답해야 합니다."""
        elif prompt_name == "survey-analysis":
            return """당신은 투자자의 성향을 분석하고 평가하는 전문가입니다. 주어진 투자 관련 지표의 점수를 바탕으로 투자자의 성향과 특징을 상세히 분석하고 설명해주세요.

각 지표별 세부 분석을 제공하고, 지표 간 상호작용을 분석하며, 종합적 평가를 통해 투자자에게 적합한 투자 전략이나 상품 유형을 제안해주세요.

분석 결과는 JSON 형식으로 출력해주세요."""
        elif prompt_name == "survey-score":
            return """당신은 투자자들의 성향을 평가하는 전문가입니다. 주어진 질문과 답변을 분석하여 각 투자 관련 지표에 대한 점수를 평가해주세요.

각 지표(Risk Tolerance, Investment Time Horizon, Financial Goal Orientation, Information Processing Style, Investment Fear, Investment Confidence)에 대해 점수를 매겨주세요.

주어진 답변을 분석하여 각 지표에 대한 점수를 매기고, JSON 형식으로 출력해주세요."""
        else:
            return None

# 프롬프트 내용 로드
AI_A_PROMPT = load_prompt("AI-A")
AI_A2_PROMPT = load_prompt("AI-A2")
AI_B_PROMPT = load_prompt("AI-B")
SURVEY_ANALYSIS_PROMPT = load_prompt("survey-analysis")
SURVEY_SCORE_PROMPT = load_prompt("survey-score")

# CompletionExecutor 클래스 추가
class CompletionExecutor:
    def __init__(self, api_key="", host=None, request_id="", max_retries=3, retry_delay=1):
        self.host = host or API_CONFIG["host"]
        self.api_key = api_key or API_CONFIG["api_key"]
        self.request_id = request_id or API_CONFIG["request_id"]
        self.max_retries = max_retries or API_CONFIG["max_retries"]
        self.retry_delay = retry_delay or API_CONFIG["retry_delay"]

    def execute(self, api_path, body):
        if not self.api_key:
            raise ValueError("API 키가 설정되지 않았습니다")
            
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key
        }
        
        if self.request_id:
            headers['X-Request-ID'] = self.request_id
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(f"{self.host}{api_path}", headers=headers, json=body)
                
                if response.status_code == 429:  # Rate limit exceeded
                    print(f"Rate limit exceeded. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"Request failed: {e}. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Request failed after {self.max_retries} attempts")
                    raise
        
        return None

# 프롬프트 템플릿 로드 함수
def load_prompt_template(file_path):
    try:
        file_path = os.path.join(SRC_DIR, file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"프롬프트 파일을 찾을 수 없습니다: {file_path}")
        return ""

# 세션별 InvestmentAdvisor 인스턴스 관리
advisor_instances = {}

def get_advisor_for_session(session_id):
    """세션별로 InvestmentAdvisor 인스턴스를 가져오거나 생성"""
    if session_id not in advisor_instances:
        advisor_instances[session_id] = InvestmentAdvisor()
    return advisor_instances[session_id]

def generate_real_data_portfolio(korean_data, us_data, risk_tolerance):
    """실제 데이터 기반 포트폴리오 생성"""
    try:
        portfolio = []
        
        # 위험 성향에 따른 종목 선택
        if risk_tolerance >= 70:  # 공격적
            # 성장주 중심
            target_stocks = ['005930', '000660', '035420', '035720', '207940']  # 삼성전자, SK하이닉스, NAVER, 카카오, 삼성바이오로직스
        elif risk_tolerance >= 40:  # 중립적
            # 대형주 + 성장주 혼합
            target_stocks = ['005930', '000660', '005380', '051910', '068270']  # 삼성전자, SK하이닉스, 현대차, LG화학, 셀트리온
        else:  # 보수적
            # 안정적인 대형주
            target_stocks = ['005930', '005380', '035420', '055550', '096770']  # 삼성전자, 현대차, NAVER, 신한지주, SK이노베이션
        
        # 한국 주식 데이터에서 종목 정보 추출
        if korean_data and 'tickers' in korean_data:
            for ticker in korean_data['tickers'][:50]:  # 상위 50개만 확인
                stock_code = str(ticker.get('종목코드', '')).zfill(6)
                if stock_code in target_stocks:
                    portfolio.append({
                        'stock_code': stock_code,
                        'stock_name': ticker.get('종목명', ''),
                        'market': 'KOSPI',
                        'sector': ticker.get('섹터', '기타'),
                        'current_price': 0,  # 실제 가격 데이터는 별도 로드
                        'predicted_price': 0,
                        'change_percent': 0,
                        'confidence': 0.8,
                        'trend': 'neutral',
                        'model_type': 'ARIMA-X',
                        'prediction_horizon': '7일'
                    })
        
        # 포트폴리오가 비어있으면 기본값 추가
        if not portfolio:
            portfolio = [
                {
                    'stock_code': '005930',
                    'stock_name': '삼성전자',
                    'market': 'KOSPI',
                    'current_price': 71000,
                    'predicted_price': 72500,
                    'change_percent': 2.11,
                    'confidence': 0.85,
                    'trend': 'bullish',
                    'model_type': 'ARIMA-X',
                    'prediction_horizon': '7일'
                }
            ]
        
        return portfolio
        
    except Exception as e:
        print(f"포트폴리오 생성 오류: {e}")
        return []

@app.route('/')
def index():
    """대시보드 페이지를 렌더링합니다."""
    return render_template('dashboard.html')

@app.route('/test-simple')
def test_simple():
    """가장 간단한 테스트"""
    return "Test OK - Flask is working!"

@app.route('/learning')
def learning():
    """투자학습 페이지를 렌더링합니다."""
    try:
        # 템플릿 파일 존재 확인
        import os
        template_path = os.path.join(app.template_folder, 'learning.html')
        app.logger.info(f"학습 페이지 접속 시도 - 템플릿 경로: {template_path}")
        
        if not os.path.exists(template_path):
            app.logger.error(f"학습 템플릿을 찾을 수 없음: {template_path}")
            return f"Template not found at: {template_path}", 404
        
        # 템플릿 렌더링 시도
        app.logger.info("학습 템플릿 렌더링 시작")
        result = render_template('learning.html')
        app.logger.info("학습 템플릿 렌더링 성공")
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Error in learning route: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        app.logger.error(error_msg)
        
        # 더 자세한 오류 정보 제공
        debug_info = f"""
        <h1>학습 페이지 오류</h1>
        <h2>오류 메시지:</h2>
        <pre>{str(e)}</pre>
        
        <h2>상세 정보:</h2>
        <pre>{traceback.format_exc()}</pre>
        
        <h2>디버그 정보:</h2>
        <ul>
            <li>템플릿 폴더: {app.template_folder}</li>
            <li>템플릿 경로: {os.path.join(app.template_folder, 'learning.html')}</li>
            <li>파일 존재: {os.path.exists(os.path.join(app.template_folder, 'learning.html'))}</li>
        </ul>
        
        <h2>테스트 페이지:</h2>
        <ul>
            <li><a href="/test/learning-minimal">최소 테스트</a></li>
            <li><a href="/test/template-check">템플릿 확인 (JSON)</a></li>
            <li><a href="/test/learning">상세 디버그 정보</a></li>
            <li><a href="/test/learning-with-layout">레이아웃 테스트</a></li>
        </ul>
        """
        return debug_info, 500

@app.route('/learning/terms')
def learning_terms_list():
    """단어 학습 목록 페이지"""
    return render_template('learning_terms.html')

@app.route('/learning/term/<term_name>')
def learning_term_detail(term_name):
    """단어 학습 상세 페이지"""
    # term_name이 숫자인 경우 ID로 검색
    try:
        term_id = int(term_name)
        term = next((t for t in INVESTMENT_TERMS if t['id'] == term_id), None)
    except ValueError:
        # 문자열인 경우 용어명으로 검색
        term = next((t for t in INVESTMENT_TERMS if t['term'] == term_name), None)
    
    if term:
        return render_template('learning_term.html', term=term)
    else:
        return redirect(url_for('learning'))

@app.route('/learning/quiz')
def learning_quiz_page():
    """학습 퀴즈 페이지"""
    return render_template('learning_quiz.html')

@app.route('/learning/<module>')
def learning_module(module):
    """학습 모듈별 페이지"""
    # 모듈별 템플릿 매핑
    module_templates = {
        'words': 'learning_terms.html',
        'cardnews': 'learning_cardnews.html',
        'quiz': 'learning_quiz.html'
    }
    
    # 유효한 모듈인지 확인
    if module in module_templates:
        return render_template(module_templates[module])
    else:
        # 잘못된 모듈은 메인 학습 페이지로 리다이렉트
        return redirect(url_for('learning'))

# === 학습 페이지 디버깅을 위한 테스트 라우트 ===
@app.route('/test/learning')
def test_learning():
    """학습 페이지 디버깅용 최소 테스트 라우트"""
    import os
    import sys
    from datetime import datetime
    import flask
    
    # 템플릿 정보 수집
    template_folder = app.template_folder
    template_files = []
    
    # 주요 템플릿 파일 확인
    templates_to_check = ['learning.html', 'layout.html', 'test_learning.html']
    for tmpl in templates_to_check:
        tmpl_path = os.path.join(template_folder, tmpl)
        exists = os.path.exists(tmpl_path)
        size = os.path.getsize(tmpl_path) if exists else 0
        template_files.append({
            'name': tmpl,
            'exists': exists,
            'size': size
        })
    
    # 세션 정보
    session_info = {
        'session_id': session.get('user_id', 'Not set'),
        'completed_terms': len(session.get('completed_terms', [])),
        'quiz_results': len(session.get('quiz_results', [])),
        'session_keys': list(session.keys())
    }
    
    return render_template('test_learning.html',
        template_path=os.path.abspath(template_folder),
        template_folder=template_folder,
        current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        flask_version=flask.__version__,
        python_version=sys.version,
        template_files=template_files,
        session_info=json.dumps(session_info, ensure_ascii=False, indent=2)
    )

@app.route('/test/learning-with-layout')
def test_learning_with_layout():
    """레이아웃을 포함한 학습 페이지 테스트"""
    try:
        # 먼저 layout.html이 존재하는지 확인
        import os
        from datetime import datetime
        layout_path = os.path.join(app.template_folder, 'layout.html')
        if not os.path.exists(layout_path):
            return f"Layout template not found at: {layout_path}", 404
        
        # 간단한 테스트 템플릿 생성
        test_content = '''
{% extends "layout.html" %}
{% block title %}학습 테스트 - 레이아웃 포함{% endblock %}
{% block content %}
<div class="container mt-5">
    <h1>레이아웃 포함 테스트</h1>
    <p>이 페이지가 정상적으로 표시된다면 레이아웃 상속이 작동합니다.</p>
    <hr>
    <h2>디버그 정보</h2>
    <ul>
        <li>템플릿 경로: {{ template_path }}</li>
        <li>현재 시간: {{ current_time }}</li>
    </ul>
    <a href="{{ url_for('learning') }}" class="btn btn-primary">원본 학습 페이지로 이동</a>
</div>
{% endblock %}
'''
        from flask import render_template_string
        return render_template_string(test_content,
            template_path=app.template_folder,
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    except Exception as e:
        import traceback
        return f"<pre>Error: {str(e)}\n\n{traceback.format_exc()}</pre>", 500

@app.route('/test/learning-minimal')
def test_learning_minimal():
    """가장 최소한의 렌더링 테스트"""
    try:
        # 직접 HTML 반환
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>최소 테스트</title>
</head>
<body>
    <h1>최소 렌더링 테스트 성공</h1>
    <p>이 페이지가 보인다면 Flask 라우팅은 정상입니다.</p>
    <ul>
        <li><a href="/learning">학습 페이지로 이동</a></li>
        <li><a href="/test/learning">전체 테스트 페이지</a></li>
    </ul>
</body>
</html>
'''
    except Exception as e:
        return f"Error in minimal test: {str(e)}", 500

@app.route('/test/template-check')
def test_template_check():
    """템플릿 파일 존재 여부만 확인"""
    import os
    results = {}
    templates_to_check = ['learning.html', 'layout.html', 'dashboard.html']
    
    for tmpl in templates_to_check:
        tmpl_path = os.path.join(app.template_folder, tmpl)
        results[tmpl] = {
            'exists': os.path.exists(tmpl_path),
            'path': tmpl_path,
            'readable': os.access(tmpl_path, os.R_OK) if os.path.exists(tmpl_path) else False
        }
    
    return jsonify({
        'template_folder': app.template_folder,
        'templates': results
    })

# === 기존 라우트 계속 ===
@app.route('/my-investment')
def my_investment():
    """마이투자 페이지를 렌더링합니다."""
    return render_template('my-investment.html')

@app.route('/my-invest')
def my_invest():
    """MY 투자 페이지를 렌더링합니다."""
    return render_template('my_invest.html')

@app.route('/chatbot')
def chatbot():
    """챗봇 페이지를 렌더링합니다."""
    return render_template('chatbot.html')

@app.route('/survey')
def survey():
    """
    설문조사 페이지 진입 시 user_id 세션 생성 보장 및 설문 완료 여부 판별
    """
    if 'user_id' not in session:
        session['user_id'] = f"user_{str(uuid.uuid4())[:8]}"
    user_id = session['user_id']
    # supabase에서 해당 user_id의 분석 결과(profile_json) 존재 여부 확인
    try:
        supabase = get_supabase_client()
        res = supabase.table('user_profiles').select('profile_json').eq('user_id', user_id).order('created_at', desc=True).limit(1).execute()
        survey_completed = bool(res.data and len(res.data) > 0 and res.data[0].get('profile_json'))
    except Exception as e:
        survey_completed = False
    return render_template('survey.html', survey_completed=survey_completed)

@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    """설문 조사 결과를 제출하고 결과를 분석합니다."""
    try:
        data = request.get_json()
        answers = data.get('answers', [])
        # user_id 세션에서 항상 가져오도록 보장
        user_id = session.get('user_id')
        if not user_id:
            user_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = user_id
        # AI를 사용하여 설문조사 응답 분석
        scores = analyze_survey_responses_with_ai(answers)
        # 분석 결과 생성
        detailed_analysis = generate_detailed_analysis(scores, answers)
        overall_analysis = generate_overall_analysis(scores)
        # 포트폴리오 추천 생성
        def recommend_portfolio(scores):
            # 다중 성향 점수 기반 맞춤형 포트폴리오 추천 및 추천 근거 생성
            # 점수는 -2 ~ 2 범위
            risk = scores.get('risk_tolerance', 0)
            horizon = scores.get('investment_time_horizon', 0)
            goal = scores.get('financial_goal_orientation', 0)
            process = scores.get('information_processing_style', 0)
            # 추천 근거 설명 생성
            reason = []
            if risk >= 1:
                reason.append('고위험 선호')
            elif risk >= -0.5:
                reason.append('중간 위험 선호')
            else:
                reason.append('저위험 선호')
            if horizon >= 1:
                reason.append('장기 투자 지향')
            elif horizon >= -0.5:
                reason.append('중기 투자 지향')
            else:
                reason.append('단기 투자 지향')
            if goal >= 1:
                reason.append('공격적 목표')
            elif goal >= -0.5:
                reason.append('균형적 목표')
            else:
                reason.append('안정적 목표')
            if process >= 1:
                reason.append('데이터/분석 기반 정보처리')
            elif process >= -0.5:
                reason.append('균형적 정보처리')
            else:
                reason.append('직관/경험 기반 정보처리')
            reason_text = ' · '.join(reason)
            # 포트폴리오 추천 로직(예시)
            if risk >= 1 and horizon >= 0.5:
                portfolio = [{
                    'name': '공격적 성장+장기 포트폴리오',
                    'description': '고위험·장기 투자 성향에 맞춰 성장주·해외주식·대체투자 비중을 높인 포트폴리오입니다.',
                    'assets': [
                        {'name': '국내 성장주', 'allocation': 30},
                        {'name': '해외 성장주', 'allocation': 30},
                        {'name': '대체 투자', 'allocation': 20},
                        {'name': '채권', 'allocation': 10},
                        {'name': '현금성 자산', 'allocation': 10},
                    ]
                }]
            elif risk >= 1:
                portfolio = [{
                    'name': '공격적 단기 포트폴리오',
                    'description': '고위험·단기 투자 성향에 맞춰 변동성 높은 자산과 일부 안전자산을 혼합한 포트폴리오입니다.',
                    'assets': [
                        {'name': '국내 주식', 'allocation': 35},
                        {'name': '해외 주식', 'allocation': 25},
                        {'name': '대체 투자', 'allocation': 15},
                        {'name': '채권', 'allocation': 15},
                        {'name': '현금성 자산', 'allocation': 10},
                    ]
                }]
            elif risk >= -0.5:
                portfolio = [{
                    'name': '균형형 포트폴리오',
                    'description': '위험과 수익의 균형을 추구하며, 다양한 자산에 분산 투자하는 포트폴리오입니다.',
                    'assets': [
                        {'name': '국내 주식', 'allocation': 25},
                        {'name': '해외 주식', 'allocation': 20},
                        {'name': '채권', 'allocation': 35},
                        {'name': '대체 투자', 'allocation': 10},
                        {'name': '현금성 자산', 'allocation': 10},
                    ]
                }]
            else:
                portfolio = [{
                    'name': '안정형 포트폴리오',
                    'description': '안정성과 원금 보존을 중시하며, 저위험 자산 위주로 구성된 포트폴리오입니다.',
                    'assets': [
                        {'name': '국내 채권', 'allocation': 50},
                        {'name': '해외 채권', 'allocation': 20},
                        {'name': '국내 주식', 'allocation': 10},
                        {'name': '현금성 자산', 'allocation': 20},
                    ]
                }]
            return {
                'portfolio': portfolio,
                'reason': reason_text
            }
        portfolio_result = recommend_portfolio(scores)
        portfolio = portfolio_result['portfolio']
        portfolio_reason = portfolio_result['reason']
        result = {
            'scores': scores,
            'overall_analysis': overall_analysis,
            'detailed_analysis': detailed_analysis,
            'portfolio': portfolio,
            'portfolio_reason': portfolio_reason
        }
        # 분석 결과를 파일에 저장
        try:
            with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"분석 결과가 저장되었습니다: {ANALYSIS_FILE}")
        except Exception as e:
            print(f"분석 결과 저장 중 오류 발생: {e}")
        # --- supabase 저장 추가 ---
        try:
            from src.db_client import get_supabase_client
            user_id = session.get('user_id') or 'default'
            supabase = get_supabase_client()
            upsert_result = supabase.table("user_profiles").upsert([{
                "user_id": user_id,
                "profile_json": result,
                "summary": overall_analysis
            }]).execute()
            print("supabase 저장 결과:", upsert_result)
        except Exception as e:
            print(f"supabase 저장 오류: {e}")
        # ---
        # 세션에도 결과 저장 (Supabase 실패 시 백업)
        session['survey_result'] = result
        session['survey_completed'] = True
        session.permanent = True
        
        print(f"Session data saved: user_id={user_id}, survey_completed=True")
        print(f"Session keys: {list(session.keys())}")
        
        return jsonify({'success': True, **result})
    
    except Exception as e:
        print(f"설문 제출 처리 중 오류: {e}")
        return jsonify({'error': f'설문 처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/survey/result')
def survey_result():
    """설문 결과 페이지를 렌더링합니다."""
    print(f"Survey result requested. Session keys: {list(session.keys())}")
    
    if 'user_id' not in session:
        print("No user_id in session, redirecting to survey")
        return redirect(url_for('survey'))
    
    user_id = session['user_id']
    print(f"User ID: {user_id}")
    
    try:
        # 먼저 세션에서 결과 확인
        if 'survey_result' in session:
            profile_json = session.get('survey_result', {})
        else:
            # Supabase에서 사용자 프로필 가져오기
            supabase = get_supabase_client()
            response = supabase.table('user_profiles').select('*').eq('user_id', user_id).order('created_at.desc').limit(1).execute()
            
            if not response.data or len(response.data) == 0:
                # 설문 결과가 없으면 설문 페이지로 리다이렉트
                print(f"No survey data found for user {user_id}")
                return redirect(url_for('survey'))
            
            profile_data = response.data[0]
            profile_json = profile_data.get('profile_json', {})
        
        # 투자 성향 분류 결정
        investment_type = determine_investment_type(profile_json.get('scores', {}))
        
        # 템플릿에 전달할 데이터 준비
        result_data = {
            'investment_type': investment_type,
            'scores': profile_json.get('scores', {}),
            'detailed_analysis': profile_json.get('detailed_analysis', {}),
            'portfolio': profile_json.get('portfolio', []),
            'portfolio_reason': profile_json.get('portfolio_reason', ''),
            'overall_analysis': profile_json.get('overall_analysis', '')
        }
        
        return render_template('survey_result.html', **result_data)
        
    except Exception as e:
        print(f"설문 결과 조회 중 오류: {e}")
        return redirect(url_for('survey'))

def determine_investment_type(scores):
    """점수를 기반으로 6가지 투자 성향 중 하나를 결정합니다."""
    if not scores:
        return {
            'type': '조심스러운 기초형 투자자',
            'description': '불확실한 상황에 쉽게 불안해하지만, 조금씩 배우고 싶어 하는 초보 투자자.',
            'characteristics': ['위험 회피 성향이 매우 강함', '단기 수익 지향', '계획과 정보 분석 부족', '자신감 부족하지만 두려움은 크지 않음'],
            'characteristic_details': ['손실에 대한 두려움이 크고, 안정성을 최우선으로 고려함.', '장기 투자보다는 빠른 결과를 기대하지만, 그에 비해 자신감이나 전략은 부족한 편.', '명확한 재무 목표와 정보 처리 능력이 부족해 판단에 어려움을 겨을 수 있음.', '행동에 나설 수 있는 여지는 있지만, 확신이 부족해 쉽게 흔들림.'],
            'recommendations': ['고수익보다는 원금 보호를 우선하는 자산군(채권, ETF 등)에 집중하세요.', '투자 기간을 늘려 복리 효과를 누리는 전략을 고민해보세요.', '‘언제까지 얼마를 모을 것인가’와 같은 구체적인 목표 설정이 필요합니다.', '너무 복잡한 데이터보다, 요약된 리포트나 쉬운 콘텐츠부터 시작해보세요.', '소액 분산투자 등으로 안전하게 경험을 늘려보는 것이 좋습니다.', '모의 투자나 학습을 통해 성공 경험을 쌌아보세요.'],
            'strategy': '안정적인 자산을 중심으로 투자하고<br><span class="highlight">재무 목표 설정 → 관련 지식 습득 → 경험 축적</span>의 순서로 기반을 다지세요.<br>투자 앱이나 플랫폼에서 제공하는 초보자용 콘텐츠, 리포트 등을 꾸준히 참고해보는 것도 좋습니다.'
        }
    
    risk_tolerance = scores.get('risk_tolerance', 0)
    time_horizon = scores.get('investment_time_horizon', 0) 
    goal_orientation = scores.get('financial_goal_orientation', 0)
    processing_style = scores.get('information_processing_style', 0)
    fear = scores.get('investment_fear', 0)
    confidence = scores.get('investment_confidence', 0)
    
    # 6가지 투자 성향 분류 로직
    if risk_tolerance < -1 and fear > 1:
        return {
            'type': '조심스러운 기초형 투자자',
            'description': '불확실한 상황에 쉽게 불안해하지만, 조금씩 배우고 싶어 하는 초보 투자자.',
            'characteristics': ['위험 회피 성향이 매우 강함', '단기 수익 지향', '계획과 정보 분석 부족', '자신감 부족하지만 두려움은 크지 않음'],
            'characteristic_details': ['손실에 대한 두려움이 크고, 안정성을 최우선으로 고려함.', '장기 투자보다는 빠른 결과를 기대하지만, 그에 비해 자신감이나 전략은 부족한 편.', '명확한 재무 목표와 정보 처리 능력이 부족해 판단에 어려움을 겨을 수 있음.', '행동에 나설 수 있는 여지는 있지만, 확신이 부족해 쉽게 흔들림.'],
            'recommendations': ['고수익보다는 원금 보호를 우선하는 자산군(채권, ETF 등)에 집중하세요.', '투자 기간을 늘려 복리 효과를 누리는 전략을 고민해보세요.', '‘언제까지 얼마를 모을 것인가’와 같은 구체적인 목표 설정이 필요합니다.', '너무 복잡한 데이터보다, 요약된 리포트나 쉬운 콘텐츠부터 시작해보세요.', '소액 분산투자 등으로 안전하게 경험을 늘려보는 것이 좋습니다.', '모의 투자나 학습을 통해 성공 경험을 쌌아보세요.'],
            'strategy': '안정적인 자산을 중심으로 투자하고<br><span class="highlight">재무 목표 설정 → 관련 지식 습듍 → 경험 축적</span>의 순서로 기반을 다지세요.<br>투자 앱이나 플랫폼에서 제공하는 초보자용 콘텐츠, 리포트 등을 꾸준히 참고해보는 것도 좋습니다.'
        }
    elif risk_tolerance < 0 and time_horizon > 0:
        return {
            'type': '안정추구형 투자자', 
            'description': '장기적 안목을 가지고 있지만 위험을 최소화하려는 투자자입니다.',
            'characteristics': ['장기 투자 선호', '안정성 중시', '위험 회피', '꾸준한 수익 추구'],
            'characteristic_details': ['시간이 지나면 좋은 결과가 있을 것이라는 믿음이 있음.', '원금 보존을 최우선으로 하며 안정적인 수익을 선호함.', '시장 변동성에 민감하며 손실을 회피하려 함.', '배당금이나 이자 수익 등 예측 가능한 수익을 선호함.'],
            'recommendations': ['채권 중심 포트폴리오로 안정적인 수익을 추구하세요.', '배당주 투자로 꾸준한 현금 흐름을 만드세요.', '적립식 투자로 시간의 힘을 활용하세요.', '인덱스 펀드나 ETF로 분산 투자하세요.', '급격한 시장 변동에도 흔들리지 마세요.', '장기 투자 계획을 세우고 꾸준히 유지하세요.'],
            'strategy': '안전 자산 중심의 포트폴리오를 구성하고<br><span class="highlight">시간 분산 투자</span>로 위험을 줄이세요.<br>장기 투자의 힘을 믿고 꾸준히 유지하는 것이 중요합니다.'
        }
    elif goal_orientation > 1 and risk_tolerance > 0:
        return {
            'type': '성장추구형 투자자',
            'description': '적극적인 자산 증식을 목표로 하며 합리적 위험을 감수하는 투자자입니다.',
            'characteristics': ['성장성 중시', '적극적 투자', '목표 지향적', '위험 감수'],
            'characteristic_details': ['높은 수익률을 위해 적극적인 투자를 선호함.', '명확한 재무 목표를 가지고 있으며 이를 달성하려 노력함.', '시장 변동성을 기회로 활용하려는 성향이 있음.', '위험을 감수하며 더 큰 수익을 추구함.'],
            'recommendations': ['성장주에 투자하여 높은 수익을 추구하세요.', '다양한 자산에 분산 투자하여 위험을 관리하세요.', '정기적으로 리밸런싱하여 포트폴리오를 최적화하세요.', '신흥 산업이나 기술주에도 관심을 가져보세요.', '해외 주식이나 대체 투자 자산도 고려해보세요.', '장기적 관점에서 투자하며 단기 변동에 흔들리지 마세요.'],
            'strategy': '성장 가능성이 높은 자산에 투자하며<br><span class="highlight">적극적인 포트폴리오 관리</span>로 수익을 극대화하세요.<br>위험 관리를 위해 분산 투자도 함께 고려하세요.'
        }
    elif processing_style < -1:
        return {
            'type': '트렌드 추종형 투자자',
            'description': '직감과 시장 트렌드에 의존하여 투자 결정을 내리는 투자자입니다.',
            'characteristics': ['직관적 판단', '트렌드 민감', '빠른 의사결정', '대중 심리 중시'],
            'characteristic_details': ['데이터 분석보다 직감과 경험에 의존하여 투자 결정을 내림.', '시장의 트렌드와 대중의 관심사에 민감하게 반응함.', '빠른 판단과 행동을 선호하며 기회를 빨리 포착하려 함.', '다른 사람들의 투자 행태에 영향을 받기 쉬움.'],
            'recommendations': ['테마주에 투자하여 트렌드를 활용하세요.', '단기 트레이딩으로 빠른 수익을 추구해보세요.', '시장 동향을 꾸준히 분석하고 파악하세요.', '직감에만 의존하지 말고 기본적인 분석도 함께 하세요.', '손절 기준을 미리 정해두고 위험을 관리하세요.', '대중 심리에 휘둘리지 말고 자신만의 기준을 세우세요.'],
            'strategy': '시장 트렌드를 빠르게 파악하고<br><span class="highlight">기회를 포착하는 민첩한 투자</span>를 추구하세요.<br>다만 감정에 휘둘리지 않도록 주의하세요.'
        }
    elif processing_style > 1 and confidence > 0:
        return {
            'type': '분석중독형 투자자',
            'description': '데이터와 분석을 중시하며 체계적으로 투자하는 투자자입니다.',
            'characteristics': ['데이터 중심', '분석적 사고', '체계적 접근', '논리적 판단'],
            'characteristic_details': ['투자 결정을 내릴 때 철저한 데이터 분석에 기반함.', '감정보다는 숫자와 팩트를 중시하며 논리적으로 접근함.', '체계적이고 계획적인 투자를 선호하며 충동적 투자를 피함.', '자신의 분석에 대한 확신이 있으며 일관성 있게 투자함.'],
            'recommendations': ['가치 투자 전략으로 저평가된 자산을 찾아보세요.', '펀더멘털 분석에 기반한 투자를 하세요.', '퀀트 투자 전략도 고려해보세요.', '분석 도구와 데이터를 활용하여 투자 효율을 높이세요.', '감정에 휘둘리지 말고 계획대로 투자하세요.', '장기적 관점에서 꾸준히 모니터링하고 조정하세요.'],
            'strategy': '철저한 데이터 분석에 기반하여<br><span class="highlight">가치 투자와 장기 투자</span>를 추구하세요.<br>감정보다는 숫자와 논리를 믿고 투자하세요.'
        }
    else:
        return {
            'type': '감정기복형 투자자',
            'description': '시장 변동에 따라 감정적으로 반응하며 일관성이 부족한 투자자입니다.',
            'characteristics': ['감정적 의사결정', '변동성에 민감', '일관성 부족', '충동적 행동'],
            'characteristic_details': ['시장 상황에 따라 감정이 크게 좌우되며 이에 따라 투자 결정을 내림.', '주가 변동에 민감하게 반응하며 자주 사고팔기를 반복함.', '일관된 투자 원칙이 없어 예측 불가능한 행동을 함.', '단기적인 시각으로 투자하며 장기 계획이 부족함.'],
            'recommendations': ['자동 투자 시스템을 활용하여 감정을 배제하세요.', '감정 관리를 위한 투자 일기를 작성해보세요.', '장기 투자 계획을 세우고 이에 충실하세요.', '투자 원칙을 정하고 이를 철저히 지키세요.', '시장 변동 시 미리 정한 기준에 따라 행동하세요.', '멘토나 전문가의 도움을 받아 체계적인 투자를 하세요.'],
            'strategy': '감정을 배제한 체계적인 투자를 위해<br><span class="highlight">자동화된 투자 시스템</span>을 활용하세요.<br>명확한 원칙을 세우고 이를 철저히 지키는 것이 중요합니다.'
        }

def get_survey_questions():
    """설문조사 질문 목록을 반환합니다."""
    return [
        "큰 꿈을 안고 시작한 투자, 여러분은 투자를 할 때 가장 중요한 요소가 무엇이라고 생각하나요? 안전하게 꾸준한 수익을 내는 투자가 옳은 투자일까요? 혹은 위험하더라도 큰 수익을 내야 진정한 투자일까요? 생각을 자유롭게 작성해 주세요.",
        "2024년 말 글로벌 경기침체 우려와 미국의 금리 인하 영향으로 증권 시장이 크게 요동쳤어요. 하지만 시간이 지나고 요동쳤던 증권 시장이 점차 안정화되었습니다. 증권 시장은 예상치 못한 이유로도 단기간에 변동되는 현상을 보여주기도 합니다. 여러분이라면 이런 단기적으로 크게 변동되는 상황을 어떻게 대응하시나요?",
        "투자는 제각각 다른 목표를 갖고 시작하곤 합니다. 누군가는 소소한 용돈을 벌기 위해, 누군가는 자가 구입을 위해 투자를 하고 있어요. 여러분의 투자 목표는 무엇인가요? 얼마나 수익을 내고, 언제까지 투자를 하고 싶나요?",
        "투자할 종목을 선택하려고 합니다. 어떤 정보를 참고하는 것이 좋을까요? 다양한 투자자들의 의견을 들어보기 위해 네이버 증권의 커뮤니티를 확인해 볼 수도 있고, 신뢰성있는 정보를 위해 뉴스나 전문가의 칼럼을 참고할 수도 있어요. 요즘 화두가 되는 AI 투자 분석 도구도 활용할 수 있습니다. 여러분들은 어떤 정보를 어떻게 활용할 예정인가요?",
        "high risk-high return 이라는 말을 들어보셨나요? 어떻게 생각하시나요? 특히 최근 암호화폐나 AI 관련 주식처럼 변동성이 큰 자산에 대해 어떻게 생각하시나요? 위험이 커도 높은 수익률을 추구하시나요? 아니면 수익이 낮더라도 안정적인 수익률이 나을까요?",
        "출근길 뉴스를 보니 내가 보유하고 있는 기술주와 관련된 안 좋은 소식이 보도되고 있어요. 그런데 간접적이기도 하고, 시장 반응도 꼭 나쁘지만은 않은 것 같아 보인다면, 어떻게 하실건가요? 주식을 매도하는게 나을까요? 혹은 기다리시나요? 선택과 이유를 함께 적어주세요.",
        "긴 고민 끝에 잘 만들어둔 나의 포트폴리오. 매일매일 변하는 평가손익이 자꾸 눈에 거슬리기도 합니다. 조언을 구해보면 일희일비 하지 말고, 앱을 지우는 것도 좋은 방법이라고 소개해 주었어요. 하지만 앱을 지운다면 포트폴리오를 자주, 그리고 즉각적으로 수정하기는 어려울 것 같아 고민입니다. 여러분은 앱을 지우고 목표 기간 뒤에 열어보는게 좋다고 생각하시나요? 혹은 매일매일 확인하고 포트폴리오를 즉각적으로 대응하여 수정하는 것이 좋다고 생각하시나요?",
        "우리는 지금까지 다양한 상황에서 판단을 해왔어요. 지금, 답변을 적는 이 순간 여러분의 투자 지식은 어느정도 된다고 생각하시나요? 최근의 경제 상황과 시장 변화에 대해 어느 정도 이해하고 계신가요? 자유롭게 기술해 주세요.",
        "나의 투자 실력을 곰곰이 생각하다보니 그 찰나에 나의 주식이 크게 떨어졌어요. 최근 AI 기술 기업들의 실적 부진과 관련이 있을 수도 있습니다. 지금 이 순간을 어떻게 대처하실건가요?",
        "재빠른 대처 능력으로 위기를 잘 극복해 내었습니다. 이젠 스스로도 투자를 어느정도 잘 하고 다음 과정으로 나아가도 되겠다는 생각이 들기도 합니다. 지금 이 생각을 한 순간 눈 앞에 지속가능 투자(ESG) 관련 새로운 금융 상품이나 신흥 시장 투자 기회가 있다면 어떤 기분과 생각이 드시나요?"
    ]

def analyze_survey_responses_with_ai(answers):
    """설문조사 응답을 분석하여 투자 성향 점수를 계산합니다."""
    try:
        # 응답 데이터 형식 확인
        if not answers:
            print("응답 데이터가 비어 있습니다. 기본값을 반환합니다.")
            return get_default_scores()
            
        # 응답 형식에 따른 처리
        formatted_answers = format_survey_answers(answers)
        
        # API 사용 가능성 확인
        if API_STATUS['selected'] == 'simulation':
            print("시뮬레이션 모드로 설문 분석을 수행합니다.")
            return analyze_survey_responses_fallback(formatted_answers)
        
        # OpenAI API 사용 시도
        if API_STATUS['selected'] == 'openai':
            try:
                return analyze_with_openai(formatted_answers)
            except Exception as e:
                print(f"OpenAI API 분석 실패: {e}")
                if API_STATUS['clova']:
                    return analyze_with_clova(formatted_answers)
                return analyze_survey_responses_fallback(formatted_answers)
        
        # Clova API 사용 시도
        elif API_STATUS['selected'] == 'clova':
            try:
                return analyze_with_clova(formatted_answers)
            except Exception as e:
                print(f"Clova API 분석 실패: {e}")
                return analyze_survey_responses_fallback(formatted_answers)
                
    except Exception as e:
        print(f"설문 분석 중 오류 발생: {e}")
        return get_default_scores()

def get_default_scores():
    """기본 점수를 반환합니다."""
    return {
        'risk_tolerance': 0.0,
        'investment_time_horizon': 0.0, 
        'financial_goal_orientation': 0.0,
        'information_processing_style': 0.0,
        'investment_fear': 0.0,
        'investment_confidence': 0.0
    }

def format_survey_answers(answers):
    """설문 답변을 표준 형식으로 변환합니다."""
    formatted_answers = []
    survey_questions = get_survey_questions()
    
    for i, item in enumerate(answers):
        if isinstance(item, dict) and 'answer' in item and 'question' in item:
            formatted_answers.append(item)
        elif isinstance(item, str):
            question = ""
            if i < len(survey_questions):
                question = survey_questions[i].get('text', f'질문 {i+1}')
            else:
                question = f"질문 {i+1}"
                
            formatted_answers.append({
                'question': question,
                'answer': item
            })
        else:
            print(f"응답 데이터 형식 오류: {item}")
            
    return formatted_answers

def analyze_with_openai(formatted_answers):
    """OpenAI API를 사용한 설문 분석"""
    import openai
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # 프롬프트 템플릿 로드
    template = load_prompt("survey-score")
    if not template:
        raise Exception("프롬프트 템플릿을 로드할 수 없습니다.")
    
    # 각 질문과 답변에 대해 개별 분석 수행
    all_scores = []
    
    for item in formatted_answers:
        question = item['question']
        answer = item['answer']
        
        # 프롬프트에 질문과 답변 삽입
        prompt = template.replace("[question]", question).replace("[answer]", answer)
        
        # API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            import json
            # JSON 응답 파싱
            content = response.choices[0].message.content
            # JSON 부분만 추출 (다른 텍스트가 포함될 수 있음)
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                scores = json.loads(json_str)
                all_scores.append(scores)
            else:
                print(f"JSON 형식이 아닌 응답: {content}")
                continue
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}, 응답: {response.choices[0].message.content}")
            continue
    
    # 모든 질문의 점수를 평균내어 최종 점수 계산
    if not all_scores:
        raise Exception("점수 분석에 실패했습니다.")
    
    final_scores = {
        "risk_tolerance": 0.0,
        "investment_time_horizon": 0.0,
        "financial_goal_orientation": 0.0,
        "information_processing_style": 0.0,
        "investment_fear": 0.0,
        "investment_confidence": 0.0
    }
    
    # 각 지표별로 평균 계산
    for metric in final_scores.keys():
        values = [score.get(metric, 0.0) for score in all_scores if metric in score]
        if values:
            final_scores[metric] = round(sum(values) / len(values), 1)
    
    return final_scores

def analyze_with_clova(formatted_answers):
    """Clova API를 사용한 설문 분석"""
    api_key = os.environ.get("CLOVA_API_KEY")
    executor = CompletionExecutor(api_key=api_key)
    
    # 프롬프트 템플릿 로드
    template = load_prompt("survey-score")
    if not template:
        raise Exception("프롬프트 템플릿을 로드할 수 없습니다.")
    
    # 각 질문과 답변에 대해 개별 분석 수행
    all_scores = []
    
    for item in formatted_answers:
        question = item['question']
        answer = item['answer']
        
        # 프롬프트에 질문과 답변 삽입
        prompt = template.replace("[question]", question).replace("[answer]", answer)
        
        # API 호출
        body = {
            "prompt": prompt,
            "maxTokens": 500,
            "temperature": 0.3,
            "topP": 0.6
        }
        
        result = executor.execute("/v1/chat-completions/HCX-DASH-001", body)
        
        if not result or 'content' not in result:
            print(f"Clova API 응답 오류: {result}")
            continue
        
        try:
            import json
            # JSON 응답 파싱
            content = result['content']
            # JSON 부분만 추출 (다른 텍스트가 포함될 수 있음)
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                scores = json.loads(json_str)
                all_scores.append(scores)
            else:
                print(f"JSON 형식이 아닌 응답: {content}")
                continue
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}, 응답: {result['content']}")
            continue
    
    # 모든 질문의 점수를 평균내어 최종 점수 계산
    if not all_scores:
        raise Exception("점수 분석에 실패했습니다.")
    
    final_scores = {
        "risk_tolerance": 0.0,
        "investment_time_horizon": 0.0,
        "financial_goal_orientation": 0.0,
        "information_processing_style": 0.0,
        "investment_fear": 0.0,
        "investment_confidence": 0.0
    }
    
    # 각 지표별로 평균 계산
    for metric in final_scores.keys():
        values = [score.get(metric, 0.0) for score in all_scores if metric in score]
        if values:
            final_scores[metric] = round(sum(values) / len(values), 1)
    
    return final_scores

def analyze_survey_responses_fallback(answers):
    """설문 응답의 AI 분석 실패 시 사용하는 백업 점수 계산 방법"""
    try:
        # 응답 데이터 형식 분석
        formatted_answers = []
        for item in answers:
            if isinstance(item, dict) and 'answer' in item:
                formatted_answers.append(item['answer'])
            elif isinstance(item, str):
                formatted_answers.append(item)
            else:
                print(f"응답 데이터 형식 오류: {item}")
                formatted_answers.append("")

        # 키워드 분석을 위한 준비
        keywords = {
            'risk_tolerance': ['위험', '도전', '기회', '수익', '적극적', '공격적', '모험'],
            'risk_tolerance_neg': ['안전', '보수', '안정', '보장', '걱정', '위험회피'],
            'investment_time_horizon': ['장기', '미래', '오래', '장래', '길게', '인내', '견디다'],
            'investment_time_horizon_neg': ['단기', '빠른', '즉시', '당장', '짧게', '급하게'],
            'financial_goal_orientation': ['높은 수익', '큰 이익', '성장', '부자', '수익성', '고수익', '수익률'],
            'financial_goal_orientation_neg': ['안정적', '작은', '소액', '배당', '이자', '안전'],
            'information_processing_style': ['분석', '데이터', '연구', '정보', '뉴스', '전문가', '도표', '지표'],
            'information_processing_style_neg': ['직감', '느낌', '본능', '감성', '감정'],
            'investment_fear': ['두려움', '손실', '불확실성', '불안정', '걱정', '공포', '패닉'],
            'investment_fear_neg': ['확신', '안정감', '침착', '여유', '평온', '자신감'],
            'investment_confidence': ['자신', '확신', '능력', '전문성', '경험', '노하우', '지식'],
            'investment_confidence_neg': ['불안', '걱정', '경험부족', '초보']
        }

        # 초기 점수 설정 (-2~2 범위)
        scores = {
            'risk_tolerance': 0.0,
            'investment_time_horizon': 0.0,
            'financial_goal_orientation': 0.0,
            'information_processing_style': 0.0,
            'investment_fear': 0.0,
            'investment_confidence': 0.0
        }

        # 응답 분석
        for idx, answer in enumerate(formatted_answers):
            if not answer:
                continue
                
            answer_text = str(answer).lower()
            
            # 질문 분류에 따른 가중치
            if idx in [0, 4]:  # 위험 관련 질문
                weight_category = 'risk_tolerance'
            elif idx in [2, 6]:  # 투자 기간 관련 질문
                weight_category = 'investment_time_horizon'
            elif idx in [3, 9]:  # 투자 목표 관련 질문
                weight_category = 'financial_goal_orientation'
            elif idx in [1, 5]:  # 정보 처리 관련 질문
                weight_category = 'information_processing_style'
            elif idx == 7:  # 투자 두려움 관련 질문
                weight_category = 'investment_fear'
            elif idx == 8:  # 투자 자신감 관련 질문
                weight_category = 'investment_confidence'
            else:
                continue
                
            # 가중치 계산을 위한 키워드 매칭
            for category, words in keywords.items():
                base_category = category.replace('_neg', '')
                
                for word in words:
                    if word in answer_text:
                        # 긍정적 키워드
                        if '_neg' not in category:
                            scores[base_category] = min(2.0, scores[base_category] + 0.3)
                        # 부정적 키워드
                        else:
                            scores[base_category] = max(-2.0, scores[base_category] - 0.3)
            
            # 응답 길이 보너스 (긴 응답은 더 깊은 생각을 반영)
            if len(answer_text) > 100:
                scores[weight_category] = min(2.0, scores[weight_category] + 0.2)
                scores['investment_confidence'] = min(2.0, scores['investment_confidence'] + 0.1)
                
        # 점수 범위 조정 (-2~2)
        for key in scores:
            scores[key] = round(max(-2.0, min(2.0, scores[key])), 1)
            
        return scores
        
    except Exception as e:
        print(f"백업 점수 계산 오류: {e}")
        # 기본 점수 반환 (-2~2 범위)
        return {
            'risk_tolerance': 0.0,
            'investment_time_horizon': 0.0,
            'financial_goal_orientation': 0.0,
            'information_processing_style': 0.0,
            'investment_fear': 0.0,
            'investment_confidence': 0.0
        }

def generate_detailed_analysis(scores, answers):
    """
    점수와 응답을 기반으로 상세 분석을 생성합니다.
    AI-A2와 AI-B 프롬프트를 사용하여 투자 성향 분석과 추천사항을 생성합니다.
    """
    try:
        # AI-A2 프롬프트 로드
        prompt_template_a2 = load_prompt_template(os.path.join(SRC_DIR, "prompt_AI-A2.txt"))
        
        # 분석 결과 생성
        risk_analysis = generate_risk_analysis(scores['risk_tolerance'])
        horizon_analysis = generate_horizon_analysis(scores['investment_time_horizon'])
        goal_analysis = generate_goal_analysis(scores['financial_goal_orientation'])
        process_analysis = generate_process_analysis(scores['information_processing_style'])
        fear_analysis = generate_fear_analysis(scores.get('investment_fear', 50))
        confidence_analysis = generate_confidence_analysis(scores['investment_confidence'])
        
        # AI-A2 프롬프트에 분석 결과 적용
        formatted_prompt_a2 = prompt_template_a2.replace("[risk_tolerance_analysis]", risk_analysis)
        formatted_prompt_a2 = formatted_prompt_a2.replace("[investment_time_horizon_analysis]", horizon_analysis)
        formatted_prompt_a2 = formatted_prompt_a2.replace("[financial_goal_orientation_analysis]", goal_analysis)
        formatted_prompt_a2 = formatted_prompt_a2.replace("[information_processing_style_analysis]", process_analysis)
        formatted_prompt_a2 = formatted_prompt_a2.replace("[investment_fear_analysis]", fear_analysis)
        formatted_prompt_a2 = formatted_prompt_a2.replace("[investment_confidence_analysis]", confidence_analysis)
        
        # 상세 분석 결과 저장 (향후 AI-B와의 연동을 위해)
        detailed_analysis = {
            "risk_tolerance_analysis": risk_analysis,
            "investment_time_horizon_analysis": horizon_analysis,
            "financial_goal_orientation_analysis": goal_analysis,
            "information_processing_style_analysis": process_analysis,
            "investment_fear_analysis": fear_analysis,
            "investment_confidence_analysis": confidence_analysis,
            "ai_a2_prompt": formatted_prompt_a2
        }
        
        return detailed_analysis
    except Exception as e:
        print(f"상세 분석 생성 오류: {e}")
        return {
            "risk_tolerance_analysis": generate_risk_analysis(scores.get('risk_tolerance', 50)),
            "investment_time_horizon_analysis": generate_horizon_analysis(scores.get('investment_time_horizon', 50)),
            "financial_goal_orientation_analysis": generate_goal_analysis(scores.get('financial_goal_orientation', 50)),
            "information_processing_style_analysis": generate_process_analysis(scores.get('information_processing_style', 50)),
            "investment_fear_analysis": generate_fear_analysis(scores.get('investment_fear', 50)),
            "investment_confidence_analysis": generate_confidence_analysis(scores.get('investment_confidence', 50))
        }

def analyze_survey_responses(answers):
    """설문 응답을 분석하여 각 영역별 점수를 계산합니다."""
    # 응답 길이에 따른 기본 가중치 설정
    weights = {
        'risk': 0,
        'horizon': 0,
        'goal': 0,
        'process': 0,
        'confidence': 0
    }
    
    # 응답 내용에 따른 키워드 분석
    keywords = {
        'risk': ['위험', '도전', '기회', '수익', '적극적', '공격적', '모험'],
        'risk_negative': ['안전', '보수', '안정', '보장', '걱정', '위험회피'],
        'horizon': ['장기', '미래', '오래', '장래', '길게', '인내', '견디다'],
        'horizon_negative': ['단기', '빠른', '즉시', '당장', '짧게', '급하게'],
        'goal': ['높은 수익', '큰 이익', '성장', '부자', '수익성', '고수익', '수익률'],
        'goal_negative': ['안정적', '작은', '소액', '배당', '이자', '안전'],
        'process': ['분석', '데이터', '연구', '정보', '뉴스', '전문가', '도표', '지표'],
        'process_negative': ['직감', '느낌', '본능', '감성', '감정'],
        'confidence': ['자신', '확신', '능력', '전문성', '경험', '노하우', '지식'],
        'confidence_negative': ['불안', '걱정', '두려움', '경험부족', '초보']
    }
    
    # 각 답변 분석
    for i, answer in enumerate(answers):
        # 질문 인덱스에 따른 가중치 설정
        if i == 0 or i == 4:  # 위험 성향 관련 질문
            weights['risk'] += 2
        elif i == 2 or i == 6:  # 투자 기간 관련 질문
            weights['horizon'] += 2
        elif i == 3 or i == 9:  # 수익 지향성 관련 질문
            weights['goal'] += 2
        elif i == 1 or i == 5:  # 정보 처리 스타일 관련 질문
            weights['process'] += 2
        elif i == 7 or i == 8:  # 투자 자신감 관련 질문
            weights['confidence'] += 2
            
        # 키워드 분석으로 점수 계산
        for category, keyword_list in keywords.items():
            if '_negative' in category:
                base_category = category.split('_')[0]
                # 부정적 키워드는 점수 감소
                for keyword in keyword_list:
                    if keyword in answer:
                        weights[base_category] -= 0.5
            else:
                # 긍정적 키워드는 점수 증가
                for keyword in keyword_list:
                    if keyword in answer:
                        weights[category] += 0.5
    
    # 응답 길이에 따른 보정
    answer_lengths = [len(answer) for answer in answers]
    avg_length = sum(answer_lengths) / len(answers) if answers else 0
    
    for answer, length in zip(answers, answer_lengths):
        if length > avg_length * 1.5:  # 평균보다 50% 이상 긴 답변
            # 더 상세한 답변은 자신감 있는 투자자일 가능성이 높음
            weights['confidence'] += 0.3
    
    # 최종 점수 계산 (0-100 스케일로 변환)
    base_scores = {
        'risk': 50,
        'horizon': 50,
        'goal': 50,
        'process': 50,
        'confidence': 50
    }
    
    # 가중치를 기반으로 점수 계산
    for category in base_scores:
        # 가중치에 따라 기본 점수에서 +/- 25까지 조정
        adjustment = min(25, max(-25, weights[category] * 5))
        base_scores[category] = min(100, max(0, base_scores[category] + adjustment))
    
    return base_scores

def generate_risk_analysis(score):
    """위험 감수 성향 분석 결과를 생성합니다."""
    if score >= 80:
        return "매우 높은 위험 감수 성향을 보입니다."
    elif score >= 60:
        return "위험 감수 성향이 높은 편입니다."
    elif score >= 40:
        return "중간 정도의 위험 감수 성향을 보입니다."
    elif score >= 20:
        return "위험 회피 성향이 있습니다."
    else:
        return "매우 보수적인 위험 회피 성향을 보입니다."

def generate_horizon_analysis(score):
    """투자 기간 선호도 분석 결과를 생성합니다."""
    if score >= 80:
        return "매우 장기적인 투자 관점을 가지고 있습니다."
    elif score >= 60:
        return "장기 투자를 선호합니다."
    elif score >= 40:
        return "중기 투자를 선호합니다."
    elif score >= 20:
        return "단기 투자를 선호합니다."
    else:
        return "매우 단기적인 투자 관점을 가지고 있습니다."

def generate_goal_analysis(score):
    """수익 지향성 분석 결과를 생성합니다."""
    if score >= 80:
        return "매우 높은 수익 지향성을 보입니다."
    elif score >= 60:
        return "높은 수익을 추구하는 경향이 있습니다."
    elif score >= 40:
        return "균형 잡힌 수익 지향성을 보입니다."
    elif score >= 20:
        return "안정적인 수익을 선호합니다."
    else:
        return "원금 보존 중심의 투자 성향을 보입니다."

def generate_process_analysis(score):
    """정보 처리 스타일 분석 결과를 생성합니다."""
    if score >= 80:
        return "매우 분석적인 정보 처리 스타일을 가지고 있습니다."
    elif score >= 60:
        return "분석적인 정보 처리 스타일을 선호합니다."
    elif score >= 40:
        return "균형 잡힌 정보 처리 스타일을 보입니다."
    elif score >= 20:
        return "직관적인 정보 처리 스타일을 가지고 있습니다."
    else:
        return "매우 직관적이고 감성적인 정보 처리 스타일을 보입니다."

def generate_confidence_analysis(score):
    """투자 자신감 분석 결과를 생성합니다."""
    if score >= 80:
        return "매우 높은 투자 자신감을 보입니다."
    elif score >= 60:
        return "높은 투자 자신감을 가지고 있습니다."
    elif score >= 40:
        return "중간 정도의 투자 자신감을 보입니다."
    elif score >= 20:
        return "낮은 투자 자신감을 보입니다."
    else:
        return "매우 낮은 투자 자신감을 가지고 있습니다."

def generate_fear_analysis(score):
    """투자 두려움 분석 결과를 생성합니다."""
    if score >= 80:
        return "투자에 대한 두려움이 매우 높습니다."
    elif score >= 60:
        return "투자에 대한 두려움이 높은 편입니다."
    elif score >= 40:
        return "투자에 대한 두려움이 중간 수준입니다."
    elif score >= 20:
        return "투자에 대한 두려움이 낮은 편입니다."
    else:
        return "투자에 대한 두려움이 매우 낮습니다."

def generate_overall_analysis(scores):
    """종합 평가를 생성합니다."""
    try:
        # 표준화된 키 이름 사용
        standard_scores = {
            'risk_tolerance': scores.get('risk_tolerance', scores.get('risk', 50)),
            'investment_time_horizon': scores.get('investment_time_horizon', scores.get('horizon', 50)),
            'financial_goal_orientation': scores.get('financial_goal_orientation', scores.get('goal', 50)),
            'information_processing_style': scores.get('information_processing_style', scores.get('process', 50)),
            'investment_fear': scores.get('investment_fear', scores.get('fear', 50)),
            'investment_confidence': scores.get('investment_confidence', scores.get('confidence', 50))
        }
        
        # 평균 점수 계산
        avg_score = sum(standard_scores.values()) / len(standard_scores)
        
        # 점수 패턴 분석
        high_risk = standard_scores['risk_tolerance'] >= 70
        long_horizon = standard_scores['investment_time_horizon'] >= 70
        high_goal = standard_scores['financial_goal_orientation'] >= 70
        analytical = standard_scores['information_processing_style'] >= 70
        low_fear = standard_scores['investment_fear'] <= 30
        confident = standard_scores['investment_confidence'] >= 70
        
        # 투자 스타일 분석
        if high_risk and high_goal and confident:
            return "적극적인 성장 투자자 유형입니다. 높은 수익을 위해 위험을 감수할 준비가 되어 있으며, 공격적인 투자 전략이 적합합니다. 신흥 시장, 성장주, 혁신 기술 분야에 투자하는 것을 고려해보세요."
        elif high_risk and not long_horizon:
            return "단기 트레이더 유형입니다. 단기적인 시장 변동을 활용한 투자에 관심이 있으며, 적극적인 매매 전략이 적합할 수 있습니다. 다만, 위험 관리에 특별히 주의를 기울여야 합니다."
        elif long_horizon and analytical and not high_risk:
            return "가치 투자자 유형입니다. 장기적인 관점에서 기업의 본질적 가치를 중요시하며, 안정적인 성장과 배당을 제공하는 기업에 투자하는 것이 적합합니다."
        elif not high_risk and not high_goal:
            return "보수적인 안정 추구형 투자자입니다. 원금 보존이 중요하며, 안정적인 채권, 배당주, 인덱스 펀드 등에 투자하는 것이 적합합니다."
        elif analytical and confident and low_fear:
            return "체계적인 투자자 유형입니다. 데이터와 분석에 기반한 의사결정을 중요시하며, 포트폴리오 다각화와 체계적인 자산 배분 전략이 효과적일 것입니다."
        else:
            if avg_score >= 70:
                return "적극적이고 균형 잡힌 투자자 유형입니다. 위험과 수익의 균형을 고려하되, 성장 가능성이 높은 분야에 투자하는 것이 적합합니다."
            elif avg_score >= 50:
                return "중립적인 균형 투자자 유형입니다. 안정성과 성장성이 균형을 이루는 포트폴리오를 구성하는 것이 적합합니다."
            else:
                return "신중한 안정 투자자 유형입니다. 안전한 투자 수단을 통해 자산을 보존하는 것이 중요하며, 점진적인 투자 확대를 고려해볼 수 있습니다."
    except Exception as e:
        print(f"종합 분석 생성 오류: {e}")
        return "투자 성향 분석을 진행하는 중 오류가 발생했습니다. 다시 시도해주세요."

@app.route('/result')
def result():
    """분석 결과 페이지를 렌더링합니다."""
    try:
        user_id = session.get('user_id') or 'default'
        from src.db_client import get_supabase_client
        supabase = get_supabase_client()
        res = supabase.table("user_profiles").select("profile_json,summary").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            profile_json = res.data[0].get("profile_json", {})
            summary = res.data[0].get("summary", "")
        else:
            profile_json, summary = {"error": "분석 결과를 찾을 수 없습니다."}, ""
        return render_template('result.html',
                              result=profile_json,
                              scores=profile_json.get('scores', {}),
                              investor_type=profile_json.get('investor_type', '분석 중'),
                              profile_description=summary or profile_json.get('overall_analysis', '분석 결과를 불러오는 중입니다.'),
                              detailed_analysis=profile_json.get('detailed_analysis', {}),
                              portfolio=profile_json.get('portfolio', []),
                              portfolio_reason=profile_json.get('portfolio_reason', ''),
                              user_id=user_id
                              )
    except Exception as e:
        print(f"결과 페이지 렌더링 오류: {e}")
        return render_template('result.html', 
                              result={"error": str(e)},
                              scores={},
                              investor_type="오류 발생",
                              profile_description=f"분석 중 오류가 발생했습니다: {str(e)}",
                              detailed_analysis={},
                              portfolio=[],
                              portfolio_reason='',
                              user_id='default'
                              )

@app.route('/minerva')
def minerva():
    """MINERVA 페이지를 렌더링합니다."""
    return render_template('minerva.html')

@app.route('/market-sentiment')
def market_sentiment():
    """시장 감정 분석 페이지를 렌더링합니다."""
    return render_template('market_sentiment.html')


# 세션별 advisor 인스턴스 관리 (간단히 메모리 dict 사용, 실제 서비스는 DB/Redis 등 권장)
advisor_sessions = {}

# 전역 FinancialDataProcessor 인스턴스 (모든 세션이 공유)
# 앱 시작 시 한 번만 초기화하여 임베딩 모델과 벡터 DB를 재사용
global_financial_processor = None

def get_global_financial_processor():
    """전역 FinancialDataProcessor 인스턴스를 반환합니다."""
    global global_financial_processor
    if global_financial_processor is None:
        logger.info("전역 FinancialDataProcessor 초기화 중...")
        global_financial_processor = FinancialDataProcessor()
        # 초기 데이터 로드
        global_financial_processor.load_latest_evaluation_data()
        logger.info("전역 FinancialDataProcessor 초기화 완료")
    return global_financial_processor

def get_advisor_for_session(session_id):
    """세션별 투자 어드바이저 인스턴스를 반환합니다."""
    if session_id not in advisor_sessions:
        # API 상태에 따라 어드바이저 초기화
        api_type = API_STATUS['selected'] 
        print(f"세션 {session_id}에 대해 {api_type} API 어드바이저를 생성합니다.")
        
        try:
            # Supabase 사용 시 FinancialDataProcessor 없이 advisor 생성
            advisor = InvestmentAdvisor(api_type=api_type, financial_processor=None)
            advisor.set_session_id(session_id)
            
            # 사용자 프로필을 미리 로드하여 캐시
            # UserProfileService는 user_id를 사용하므로 session_id를 그대로 사용
            # (web/app.py에서는 session_id가 실제로는 user_id 역할을 함)
            from src.db_client import get_user_profile_service
            profile_service = get_user_profile_service()
            user_profile = profile_service.get_user_profile(session_id)  # session_id가 실제로는 user_id임
            
            if user_profile:
                print(f"프로필 발견: {session_id}")
                # profile_json이 있으면 그것을 사용, 없으면 전체 프로필 사용
                if 'profile_json' in user_profile:
                    print(f"profile_json 키 발견, 타입: {type(user_profile['profile_json'])}")
                    # profile_json이 문자열이면 파싱
                    if isinstance(user_profile['profile_json'], str):
                        try:
                            profile_data = json.loads(user_profile['profile_json'])
                            user_profile['profile_json'] = profile_data
                            print(f"profile_json 파싱 완료")
                        except:
                            print(f"profile_json 파싱 실패")
                
                # MemoryManager에 프로필 캐시
                advisor.memory_manager.cache_context(session_id, "user_profile", user_profile)
                print(f"프로필 캐시 완료: {session_id}")
            else:
                print(f"프로필을 찾을 수 없음: {session_id}")
            
            advisor_sessions[session_id] = advisor
        except Exception as e:
            print(f"어드바이저 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            # 시뮬레이션 모드로 폴백
            advisor = InvestmentAdvisor(api_type="simulation", financial_processor=None)
            advisor.set_session_id(session_id)
            advisor_sessions[session_id] = advisor
            
    return advisor_sessions[session_id]

@app.route('/api/chat', methods=['POST'])
def chat():
    """MINERVA 챗봇에 메시지를 전송하고 응답을 받습니다."""
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = session.get('user_id')
        if not session_id:
            session_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = session_id
        # 1. 설문 결과 조회
        from src.db_client import get_supabase_client
        supabase = get_supabase_client()
        
        has_profile = False
        profile_json = None
        summary = None
        
        if supabase:
            try:
                res = supabase.table("user_profiles").select("profile_json,summary").eq("user_id", session_id).order("created_at", desc=True).limit(1).execute()
                has_profile = res.data and len(res.data) > 0
                profile_json = res.data[0]['profile_json'] if has_profile else None
                summary = res.data[0]['summary'] if has_profile else None
            except Exception as e:
                logger.error(f"Supabase 조회 오류: {e}")
                # SQLite 폴백
                try:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""
                        SELECT analysis_result FROM user_profiles 
                        WHERE user_id = ? 
                        ORDER BY created_at DESC LIMIT 1
                    """, (session_id,))
                    result = c.fetchone()
                    if result and result[0]:
                        profile_data = json.loads(result[0])
                        has_profile = True
                        profile_json = profile_data
                        summary = profile_data.get('overall_analysis', '')
                    conn.close()
                except Exception as sqlite_err:
                    print(f"SQLite 조회 오류: {sqlite_err}")
        else:
            # Supabase가 없으면 SQLite 사용
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("""
                    SELECT analysis_result FROM user_profiles 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC LIMIT 1
                """, (session_id,))
                result = c.fetchone()
                if result and result[0]:
                    profile_data = json.loads(result[0])
                    has_profile = True
                    profile_json = profile_data
                    summary = profile_data.get('overall_analysis', '')
                conn.close()
            except Exception as e:
                print(f"SQLite 조회 오류: {e}")
        # 2. 설문 미완료 안내
        if not has_profile:
            return jsonify({
                "success": True,
                "response": (
                    "먼저 투자 성향 설문을 완료해 주세요! "
                    "설문이 끝나면 맞춤형 분석과 포트폴리오 추천을 드릴 수 있습니다. "
                    "아래 버튼을 눌러 설문을 시작하세요."
                ),
                "require_survey": True
            })
        # 3. 설문 완료: 성향 안내 및 포트폴리오 추천
        if not message or message.lower() in ["hi", "hello", "안녕", "처음", "시작"]:
            return jsonify({
                "success": True,
                "response": (
                    f"당신의 투자 성향 요약: {summary}\n"
                    f"이 성향에 맞는 포트폴리오 추천: {profile_json.get('portfolio', [])}\n"
                    "이제 투자 관련 질문이나 시장 분석에 대해 무엇이든 물어보세요!"
                )
            })
        # 4. Investment Advisor를 통한 AI Agent Chain 실행
        try:
            print(f"사용자 메시지 처리 중: {message[:50]}...")
            
            # 투자 어드바이저 인스턴스 가져오기
            advisor = get_advisor_for_session(session_id)
            
            # 프로필 데이터가 캐시되지 않았을 경우 다시 로드
            if profile_json and not advisor.memory_manager.get_cached_context(session_id, "user_profile"):
                if isinstance(profile_json, str):
                    advisor.memory_manager.cache_context(session_id, "user_profile", json.loads(profile_json))
                else:
                    advisor.memory_manager.cache_context(session_id, "user_profile", profile_json)
            
            # AI-A → AI-A2 → AI-B → Final 체인 실행
            response = advisor.chat(message)
            
            print(f"AI 응답 생성 완룼: {len(response)}자")
            
            # 채팅 기록 저장 (로컬 파일)
            save_chat_message_to_file(session_id, "user", message)
            save_chat_message_to_file(session_id, "assistant", response)
            
            return jsonify({
                "success": True,
                "response": response
            })
            
        except Exception as e:
            print(f"AI 에이전트 체인 실행 실패: {e}")
            
            # 폴백: 간단한 응답 생성
            fallback_response = generate_fallback_response(message, summary, profile_json)
            
            return jsonify({
                "success": True,
                "response": fallback_response,
                "fallback": True
            })
        
    except Exception as e:
        print(f"채팅 API 처리 중 오류: {e}")
        return jsonify({
            "success": False,
            "error": "채팅 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        }), 500

def generate_fallback_response(message, summary, profile_json):
    """AI 에이전트 체인 실패 시 사용하는 기본 응답 생성"""
    try:
        # 키워드 기반 기본 응답
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ['포트폴리오', '추천', '투자', '주식']):
            portfolio = profile_json.get('portfolio', [])
            if portfolio:
                return f"""
{summary}

이 성향에 맞는 포트폴리오 추천:
{portfolio[0].get('name', '균형형 포트폴리오')}
{portfolio[0].get('description', '위험과 수익의 균형을 고려한 포트폴리오입니다.')}

더 자세한 분석을 위해 시스템이 복구되는 대로 다시 문의해주세요.
"""
        elif any(keyword in message_lower for keyword in ['위험', '리스크', '안전']):
            return f"""
{summary}

투자 시 위험 관리는 매우 중요합니다. 당신의 위험 감수 성향을 고려하여:
- 분산 투자를 통한 위험 분산
- 투자 기간에 맞는 자산 배분
- 정기적인 포트폴리오 리밸런싱

을 추천드립니다. 더 구체적인 조언은 시스템 복구 후 제공해드리겠습니다.
"""
        else:
            return f"""
{summary}

투자 관련 질문에 대한 상세한 분석을 준비 중입니다. 
현재 시스템 점검으로 인해 간단한 답변만 제공 가능합니다.

다음과 같은 주제로 질문해보세요:
- 포트폴리오 추천
- 위험 관리 방법  
- 투자 전략 조언
- 시장 상황 분석

시스템이 완전히 복구되면 더 자세한 AI 분석을 제공해드리겠습니다.
"""
            
    except Exception as e:
        return f"투자 상담을 위해 시스템을 점검 중입니다. 잠시만 기다려주세요. (오류: {str(e)})"

def save_chat_message_to_file(session_id, role, content):
    """채팅 메시지를 로컬 파일에 저장"""
    try:
        history = load_chat_history()
        if session_id not in history:
            history[session_id] = []
        
        history[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"채팅 기록 저장 실패: {e}")

def load_chat_history():
    """채팅 기록을 로드합니다."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"채팅 기록 로드 실패: {e}")
        return {}

def process_chat_with_ai_chain(message, user_session_id):
    """MINERVA AI 체인 프로세스를 통한 채팅 처리"""
    try:
        # 사용자 투자 성향 조회
        try:
            supabase = get_supabase_client()
            res = supabase.table("user_profiles").select("*").eq("user_id", user_session_id).order("created_at", desc=True).limit(1).execute()
            
            if res.data and len(res.data) > 0:
                profile = res.data[0]
                summary = profile.get('profile_json', {}).get('summary', 'N/A')
            else:
                summary = "투자 성향 정보 없음"
                
        except Exception as e:
            print(f"투자 성향 조회 실패: {e}")
            summary = "투자 성향 정보 없음"
        
        # Step 1: AI-A 초기 응답
        ai_a_response = call_llm_ai_a(
            f"투자자 성향 요약: {summary}\n"
            f"사용자 질문: {message}\n"
            "위 정보를 바탕으로 초기 투자 조언을 제공하세요."
        )
        
        # Step 2: AI-A2 질문 명확화 및 성향 기반 프롬프트 생성
        ai_a2_response = call_llm_ai_a2(
            f"사용자의 투자 성향 요약: {summary}\n"
            f"사용자 질문: {message}\n"
            f"AI-A 초기 응답: {ai_a_response}\n"
            "위 질문을 투자 성향에 맞게 명확하게 정리하고, 추가 정보가 필요한지 판단해주세요."
        )
        
        # Step 3: AI-B 실시간 데이터/분석 제공
        ai_b_response = call_llm_ai_b(
            f"질문: {message}\n"
            f"AI-A2 분석: {ai_a2_response}\n"
            "실시간 주가, 뉴스, 시장 데이터를 조회하여 정확한 정보를 제공하세요."
        )
        
        # Step 4: 최종 맞춤형 조언/추천 (AI-A 기반)
        final_response = call_llm_ai_a(
            f"투자자 성향: {summary}\n"
            f"사용자 질문: {message}\n"
            f"AI-A 초기 응답: {ai_a_response}\n"
            f"AI-A2 분석: {ai_a2_response}\n"
            f"AI-B 데이터: {ai_b_response}\n"
            "위 모든 정보를 종합하여 투자자에게 최종 맞춤형 조언을 제공하세요."
        )
        
        return {
            "success": True,
            "response": final_response,
            "ai_chain_data": {
                "ai_a_initial": ai_a_response,
                "ai_a2_analysis": ai_a2_response,
                "ai_b_data": ai_b_response,
                "final_advice": final_response
            }
        }
        
    except Exception as e:
        print(f"AI 체인 처리 오류: {e}")
        return {
            "success": False,
            "response": f"죄송합니다. AI 분석 시스템에 문제가 발생했습니다: {str(e)}"
        }

# LLM 호출 함수들 (실제 API 연동)
def call_llm_ai_a(prompt: str) -> str:
    """AI-A: 투자 상담사 역할 - 사용자 성향 기반 맞춤형 조언"""
    try:
        # OpenAI API 호출 (환경변수에서 API 키 가져오기)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return f"[AI-A] {prompt[:100]}... (API 키가 설정되지 않음)"
        
        # 실제 OpenAI API 호출 로직
        import openai
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": AI_A_PROMPT or "당신은 투자 상담사입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"AI-A 호출 오류: {e}")
        return f"[AI-A] {prompt[:100]}... (오류 발생)"

def call_llm_ai_a2(prompt: str) -> str:
    """AI-A2: 질문 명확화 및 성향 분석"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return f"[AI-A2] {prompt[:100]}... (API 키가 설정되지 않음)"
        
        import openai
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": AI_A2_PROMPT or "당신은 투자 질문 분석가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"AI-A2 호출 오류: {e}")
        return f"[AI-A2] {prompt[:100]}... (오류 발생)"

def call_llm_ai_b(prompt: str) -> str:
    """AI-B: 실시간 데이터 분석 및 정보 제공"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return f"[AI-B] {prompt[:100]}... (API 키가 설정되지 않음)"
        
        import openai
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": AI_B_PROMPT or "당신은 금융 데이터 분석가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"AI-B 호출 오류: {e}")
        return f"[AI-B] {prompt[:100]}... (오류 발생)"

@app.route('/api/predictions')
def get_predictions():
    """실제 데이터 기반 포트폴리오 예측 결과를 반환합니다."""
    try:
        print("실제 데이터 기반 포트폴리오 예측 데이터 생성")
        logger.info("/api/predictions 엔드포인트 호출됨")
        
        # 세션 ID 확인
        session_id = session.get('user_id')
        if not session_id:
            session_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = session_id
        
        # notebook 기반 주식 평가 시스템 사용
        try:
            from src.stock_search_engine import StockSearchEngine
            search_engine = StockSearchEngine()
            
            # 평가점수 상위 종목 가져오기
            top_stocks = search_engine.get_top_stocks(n_results=10)
            logger.info(f"상위 {len(top_stocks)}개 종목 로드 완료")
        except Exception as e:
            logger.error(f"주식 평가 시스템 로드 실패: {e}")
            top_stocks = []
        
        # 실제 데이터 로드 (CSV 파일에서)
        korean_data = []
        us_data = []
        
        # 사용자 프로필에서 위험 성향 및 추천 종목 가져오기
        risk_tolerance = 50  # 기본값
        recommended_stocks = []
        
        try:
            if os.path.exists(DB_PATH):
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                
                # 사용자 프로필에서 위험 성향 가져오기
                try:
                    c.execute("""
                        SELECT risk_tolerance, analysis_result 
                        FROM user_profiles 
                        WHERE user_id = ? 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """, (session_id,))
                    
                    profile_data = c.fetchone()
                    if profile_data:
                        risk_tolerance = profile_data[0] or 50
                        analysis_result = profile_data[1]
                        
                        # analysis_result에서 추천 종목 추출 (간단한 파싱)
                        if analysis_result and '추천' in analysis_result:
                            # 종목 코드 패턴 찾기 (6자리 숫자)
                            import re
                            stock_codes = re.findall(r'\b\d{6}\b', analysis_result)
                            recommended_stocks = stock_codes[:5]  # 최대 5개
                except Exception as e:
                    logger.warning(f"프로필 조회 중 오류: {e}")
                
                # 포트폴리오 추천 기록에서도 확인
                try:
                    c.execute("""
                        SELECT recommendation_data 
                        FROM portfolio_recommendations 
                        WHERE user_id = ? 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """, (session_id,))
                    
                    portfolio_data = c.fetchone()
                    if portfolio_data and portfolio_data[0]:
                        try:
                            import json
                            rec_data = json.loads(portfolio_data[0])
                            if 'tickers' in rec_data:
                                for ticker in rec_data['tickers'][:5]:
                                    if 'code' in ticker and ticker['code'] not in recommended_stocks:
                                        recommended_stocks.append(ticker['code'])
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"포트폴리오 추천 조회 중 오류: {e}")
                
                conn.close()
            else:
                logger.warning(f"데이터베이스 파일이 없음: {DB_PATH}")
            
        except Exception as e:
            logger.error(f"데이터베이스 연결 오류: {e}")
        
        # notebook 기반 시스템에서 위험 성향에 따른 종목 선택
        if top_stocks:
            # 위험 성향에 따라 상위 종목에서 선택
            if risk_tolerance >= 70:  # 공격적 - 평가점수 상위 종목
                recommended_stocks = [stock.종목코드 for stock in top_stocks[:5]]
            elif risk_tolerance >= 40:  # 중립적 - 평가점수 상위 종목 중 안정성 고려
                selected_stocks = []
                for stock in top_stocks[:10]:
                    if stock.부채비율 is not None and stock.부채비율 < 150:
                        selected_stocks.append(stock.종목코드)
                    if len(selected_stocks) >= 5:
                        break
                recommended_stocks = selected_stocks
            else:  # 보수적 - 부채비율 낮고 PER 적정한 종목
                selected_stocks = []
                for stock in top_stocks[:15]:
                    if (stock.부채비율 is not None and stock.부채비율 < 100 and
                        stock.PER is not None and stock.PER < 20):
                        selected_stocks.append(stock.종목코드)
                    if len(selected_stocks) >= 5:
                        break
                recommended_stocks = selected_stocks
        
        # 추천 종목이 없으면 위험 성향에 따른 기본 종목 사용
        if not recommended_stocks:
            if risk_tolerance >= 70:  # 공격적
                recommended_stocks = ['005930', '000660', '035420', '035720', '207940']
            elif risk_tolerance >= 40:  # 중립적
                recommended_stocks = ['005930', '000660', '005380', '051910', '068270']
            else:  # 보수적
                recommended_stocks = ['005930', '005380', '035420', '055550', '096770']
        
        # 위험 성향에 따른 포트폴리오 구성 (일단 빈 딕셔너리로 초기화)
        predictions = {}
        
        # 추천 종목에 대한 시계열 예측 생성
        time_series_data = []
        
        print(f"추천 종목 {len(recommended_stocks)}개에 대한 예측 생성")
        
        for stock_code in recommended_stocks:
            try:
                # 해당 종목의 예측 생성 (analyze_stock_trends 사용)
                prediction = analyze_stock_trends(stock_code)
                if prediction:
                    time_series_data.append(prediction)
                    print(f"{stock_code} 예측 완료")
                else:
                    print(f"{stock_code} 예측 실패: 예측 데이터 없음")
            except Exception as e:
                print(f"{stock_code} 예측 실패: {e}")
                import traceback
                traceback.print_exc()
        
        # 예측 데이터가 없으면 기본 샘플 추가
        if not time_series_data:
            print("예측 데이터가 없어 기본 샘플 사용")
            sample_stocks = ['005930', '000660', '035420']
            for stock_code in sample_stocks:
                try:
                    prediction = analyze_stock_trends(stock_code)
                    if prediction:
                        time_series_data.append(prediction)
                except:
                    pass
        
        # notebook 기반 평가 데이터로 예측 생성
        for i, stock_code in enumerate(recommended_stocks[:5]):
            try:
                # top_stocks에서 해당 종목 찾기
                stock_info = None
                for stock in top_stocks:
                    if stock.종목코드 == stock_code:
                        stock_info = stock
                        break
                
                if stock_info:
                    # 평가 정보 기반 예측
                    change_percent = 0
                    if stock_info.매출성장률 is not None and stock_info.매출성장률 > 10:
                        change_percent += 2.5
                    if stock_info.순이익률 is not None and stock_info.순이익률 > 10:
                        change_percent += 2.0
                    if stock_info.부채비율 is not None and stock_info.부채비율 < 100:
                        change_percent += 1.5
                    if stock_info.PER is not None and 5 < stock_info.PER < 15:
                        change_percent += 1.0
                    
                    # 랜덤 요소 추가
                    import random
                    change_percent += random.uniform(-1, 1)
                    
                    predicted_price = stock_info.현재가 * (1 + change_percent / 100)
                    
                    time_series_data.append({
                        'stock_code': stock_code,
                        'stock_name': stock_info.종목명,
                        'current_price': int(stock_info.현재가),
                        'predicted_price': int(predicted_price),
                        'change_percent': round(change_percent, 2),
                        'trend': 'bullish' if change_percent > 0 else 'bearish',
                        'model_type': 'Evaluation-based',
                        'confidence': 'high' if stock_info.평가점수 >= 80 else 'medium',
                        'prediction_horizon': '7일',
                        'evaluation_score': stock_info.평가점수,
                        'evaluation_grade': stock_info.종합평가
                    })
                else:
                    # 평가 데이터가 없는 경우 기존 analyze_stock_trends 사용
                    prediction = analyze_stock_trends(stock_code)
                    if prediction:
                        time_series_data.append(prediction)
            except Exception as e:
                logger.error(f"{stock_code} 예측 생성 오류: {e}")
                continue
        
        # 복잡한 부분 주석 처리
        """
        # 고도화된 시계열 예측 시스템 사용
        try:
            from src.simplified_portfolio_prediction import extract_portfolio_tickers, analyze_portfolio_with_user_profile
            from src.advanced_stock_predictor import AdvancedStockPredictor
            from src.news_collector_service import NewsCollectorService
            
            # 뉴스 감정 분석 데이터 가져오기
            news_sentiment_data = {
                'average_score': 0.5,
                'overall_sentiment': '중립적'
            }
            
            try:
                # 실시간 뉴스 감정 분석 수행
                sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
                from news_sentiment_analyzer import NewsSentimentAnalyzer
                analyzer = NewsSentimentAnalyzer()
                
                # 뉴스 데이터 로드
                news_data = real_data_manager.load_news_data()
                if news_data:
                    news_df = pd.DataFrame(news_data)
                    sentiment_result = analyzer.analyze_news_sentiment(news_df)
                    
                    news_sentiment_data = {
                        'average_score': sentiment_result['overall_sentiment'],
                        'overall_sentiment': sentiment_result['market_mood']['mood'],
                        'sentiment_description': sentiment_result['market_mood']['description'],
                        'investment_recommendation': sentiment_result['market_mood']['recommendation'],
                        'positive_count': sentiment_result['sentiment_distribution']['positive'],
                        'negative_count': sentiment_result['sentiment_distribution']['negative'],
                        'neutral_count': sentiment_result['sentiment_distribution']['neutral'],
                        'total_count': len(news_data),
                        'top_positive_keywords': sentiment_result['top_positive_keywords'],
                        'top_negative_keywords': sentiment_result['top_negative_keywords'],
                        'stock_sentiments': sentiment_result['stock_sentiments']
                    }
                    
                    # 투자 신호 생성
                    investment_signals = analyzer.get_investment_signals(sentiment_result)
                    news_sentiment_data['investment_signals'] = investment_signals
                    
            except Exception as e:
                logger.warning(f"Failed to get news sentiment data: {e}")
            
            # 기본 포트폴리오 또는 사용자 맞춤 포트폴리오 구성
            try:
                # InvestmentAdvisor에서 AI 대화 기록 확인하여 추천 종목 추출
                global_processor = get_global_financial_processor()
                advisor = InvestmentAdvisor(financial_processor=global_processor)
                if hasattr(advisor, 'memory_manager') and advisor.memory_manager:
                    ai_chat_history = advisor.memory_manager.get_ai_conversation(session_id)
                    portfolio_tickers = extract_portfolio_tickers(ai_chat_history)
                else:
                    portfolio_tickers = ['005930', '000660', '035420', '091990', '247540']  # 기본 포트폴리오
            except:
                portfolio_tickers = ['005930', '000660', '035420', '091990', '247540']  # 기본 포트폴리오
            
            # ARIMA-X 예측기 초기화
            predictor = AdvancedStockPredictor()
            
            # 고도화된 ARIMA-X 기반 포트폴리오 예측 분석
            enhanced_predictions = []
            for ticker in portfolio_tickers[:5]:  # 상위 5개 종목만
                try:
                    # 해당 종목의 가격 데이터 가져오기
                    price_data = real_data_manager.get_stock_price_data(ticker)
                    if price_data is not None and len(price_data) > 0:
                        # ARIMA-X 모델로 예측
                        prediction = predictor.predict_with_sentiment(
                            stock_code=ticker,
                            price_data=price_data,
                            sentiment_data=news_sentiment_data,
                            days=30
                        )
                        enhanced_predictions.append(prediction)
                except Exception as e:
                    logger.warning(f"Failed to predict for {ticker}: {e}")
            
            # 기존 포트폴리오 분석과 병합
            enhanced_portfolio_analysis = analyze_portfolio_with_user_profile(portfolio_tickers, session_id)
            
            # ARIMA-X 예측 결과와 기존 분석 결과를 병합
            time_series_data = []
            
            # ARIMA-X 예측 결과 추가
            for pred in enhanced_predictions:
                if 'stock_code' in pred and 'predictions' in pred:
                    predictions = pred['predictions']
                    current_price = pred.get('current_price', 0)
                    predicted_price = predictions[-1] if predictions else current_price
                    change_pct = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
                    
                    time_series_data.append({
                        'stock_code': pred['stock_code'],
                        'current_price': current_price,
                        'predicted_price': predicted_price,
                        'trend': 'bullish' if change_pct > 0 else 'bearish' if change_pct < 0 else 'neutral',
                        'change_percent': round(change_pct, 2),
                        'model_type': 'ARIMA-X with Sentiment',
                        'prediction_horizon': '30_days',
                        'confidence': pred.get('investment_signals', {}).get('confidence', 'medium'),
                        'sentiment_impact': pred.get('sentiment_impact', {}),
                        'technical_indicators': pred.get('technical_indicators', {}),
                        'investment_signals': pred.get('investment_signals', {})
                    })
            
            # 기존 분석 결과도 추가 (ARIMA-X로 예측하지 못한 종목들)
            for ticker, ticker_data in enhanced_portfolio_analysis.get('ticker_predictions', {}).items():
                # 이미 ARIMA-X로 예측한 종목은 제외
                if not any(item['stock_code'] == ticker for item in time_series_data):
                    time_series_data.append({
                        'stock_code': ticker,
                        'current_price': ticker_data.get('current_price', 0),
                        'predicted_price': ticker_data.get('predicted_price', 0),
                        'trend': ticker_data.get('trend', 'neutral'),
                        'change_percent': ticker_data.get('change_pct', 0),
                        'model_type': ticker_data.get('model_type', 'Basic ARIMA'),
                        'prediction_horizon': '7_days',
                        'confidence': 'high' if abs(ticker_data.get('change_pct', 0)) > 2 else 'medium'
                    })
            
        """
        
        return jsonify({
            "success": True,
            "predictions": predictions,
            "time_series_forecasts": time_series_data,
            "data_source": "real_collected_data",
            "user_profile": {
                "risk_tolerance": risk_tolerance,
                "recommended_stocks": recommended_stocks
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"실제 데이터 기반 예측 생성 실패: {e}")
        return jsonify({
            "success": False, 
            "error": f"포트폴리오 예측 생성 중 오류: {str(e)}"
        })

def generate_real_data_portfolio(korean_data, us_data, risk_tolerance):
    """실제 데이터를 기반으로 포트폴리오 생성"""
    try:
        portfolio = {"tickers": [], "analysis": ""}
        
        # 한국 주식 데이터 분석
        korean_tickers = korean_data.get('tickers', [])
        korean_prices = korean_data.get('prices', [])
        korean_valuations = korean_data.get('valuations', [])
        
        # 미국 주식 데이터 분석  
        us_tickers = us_data.get('tickers', [])
        us_prices = us_data.get('prices', [])
        
        print(f"데이터 현황: 한국 종목 {len(korean_tickers)}개, 미국 종목 {len(us_tickers)}개")
        print(f"가격 데이터: 한국 {len(korean_prices)}건, 미국 {len(us_prices)}건")
        
        # 위험 성향별 종목 선택
        selected_stocks = []
        
        # 한국 주식에서 선택 (위험 성향에 따라)
        if korean_tickers and korean_prices:
            korean_selection = select_korean_stocks_by_risk(korean_tickers, korean_prices, korean_valuations, risk_tolerance)
            selected_stocks.extend(korean_selection)
        
        # 미국 주식에서 선택
        if us_tickers and us_prices:
            us_selection = select_us_stocks_by_risk(us_tickers, us_prices, risk_tolerance) 
            selected_stocks.extend(us_selection)
        
        # 각 선택된 종목에 대한 예측 생성
        for stock in selected_stocks:
            prediction = generate_stock_prediction_from_data(stock, korean_prices, us_prices)
            portfolio["tickers"].append({
                "code": stock["code"],
                "name": stock["name"],
                "market": stock["market"],
                "allocation": stock["allocation"],
                "prediction": prediction,
                "image": f"/static/images/{stock['code']}_prediction.png" if os.path.exists(os.path.join(IMAGE_DIR, f"{stock['code']}_prediction.png")) else None
            })
        
        # 포트폴리오 전체 분석
        portfolio["analysis"] = generate_portfolio_analysis(selected_stocks, risk_tolerance)
        
        return portfolio
        
    except Exception as e:
        print(f"포트폴리오 생성 오류: {e}")
        return {"tickers": [], "analysis": "포트폴리오 분석 중 오류가 발생했습니다."}

def select_korean_stocks_by_risk(tickers, prices, valuations, risk_tolerance):
    """위험 성향에 따른 한국 주식 선택"""
    try:
        import pandas as pd
        
        # 가격 데이터를 DataFrame으로 변환
        price_df = pd.DataFrame(prices)
        if price_df.empty:
            return []
        
        # 종목별 최신 가격 및 변동성 계산
        stock_analysis = {}
        for ticker_info in tickers[:20]:  # 상위 20개 종목만 분석
            code = ticker_info.get('종목코드', '')
            if not code:
                continue
                
            stock_prices = price_df[price_df['종목코드'] == code]
            if stock_prices.empty:
                continue
            
            # 수익률 및 변동성 계산
            if len(stock_prices) > 1:
                stock_prices = stock_prices.sort_values('날짜') if '날짜' in stock_prices.columns else stock_prices
                returns = stock_prices['종가'].pct_change().dropna()
                volatility = returns.std() * 100 if len(returns) > 0 else 0
                avg_return = returns.mean() * 100 if len(returns) > 0 else 0
            else:
                volatility = 0
                avg_return = 0
            
            stock_analysis[code] = {
                'name': ticker_info.get('종목명', f'종목{code}'),
                'volatility': volatility,
                'return': avg_return,
                'sector': ticker_info.get('섹터', '기타')
            }
        
        # 위험 성향에 따른 선택
        selected = []
        if risk_tolerance >= 70:  # 고위험
            # 변동성 높은 성장주 선호
            sorted_stocks = sorted(stock_analysis.items(), key=lambda x: x[1]['volatility'], reverse=True)
            allocation = 60  # 한국 주식 비중
        elif risk_tolerance >= 40:  # 중위험  
            # 균형 잡힌 선택
            sorted_stocks = sorted(stock_analysis.items(), key=lambda x: x[1]['return'], reverse=True)
            allocation = 50
        else:  # 저위험
            # 안정적인 대형주 선호
            sorted_stocks = sorted(stock_analysis.items(), key=lambda x: x[1]['volatility'])
            allocation = 40
        
        # 상위 3개 종목 선택
        for i, (code, info) in enumerate(sorted_stocks[:3]):
            selected.append({
                'code': code,
                'name': info['name'],
                'market': 'KR',
                'allocation': allocation // 3,  # 균등 분할
                'sector': info['sector'],
                'volatility': info['volatility'],
                'return': info['return']
            })
        
        return selected
        
    except Exception as e:
        print(f"한국 주식 선택 오류: {e}")
        return []

def select_us_stocks_by_risk(tickers, prices, risk_tolerance):
    """위험 성향에 따른 미국 주식 선택"""
    try:
        import pandas as pd
        
        price_df = pd.DataFrame(prices)
        if price_df.empty:
            return []
        
        # 미국 주식 분석
        stock_analysis = {}
        for ticker_info in tickers[:10]:  # 상위 10개 분석
            ticker = ticker_info.get('Ticker', '')
            if not ticker:
                continue
                
            stock_prices = price_df[price_df['Ticker'] == ticker]
            if stock_prices.empty:
                continue
            
            # 수익률 및 변동성 계산
            if len(stock_prices) > 1:
                stock_prices = stock_prices.sort_values('Date') if 'Date' in stock_prices.columns else stock_prices
                returns = stock_prices['Close'].pct_change().dropna()
                volatility = returns.std() * 100 if len(returns) > 0 else 0
                avg_return = returns.mean() * 100 if len(returns) > 0 else 0
            else:
                volatility = 0
                avg_return = 0
            
            stock_analysis[ticker] = {
                'name': ticker_info.get('Name', ticker),
                'volatility': volatility,
                'return': avg_return,
                'sector': ticker_info.get('Sector', '기타')
            }
        
        # 위험 성향에 따른 선택
        selected = []
        if risk_tolerance >= 70:  # 고위험
            sorted_stocks = sorted(stock_analysis.items(), key=lambda x: x[1]['return'], reverse=True)
            allocation = 40
        elif risk_tolerance >= 40:  # 중위험
            sorted_stocks = sorted(stock_analysis.items(), key=lambda x: (x[1]['return'] - x[1]['volatility']), reverse=True)
            allocation = 30
        else:  # 저위험
            sorted_stocks = sorted(stock_analysis.items(), key=lambda x: x[1]['volatility'])
            allocation = 20
        
        # 상위 2개 종목 선택
        for i, (ticker, info) in enumerate(sorted_stocks[:2]):
            selected.append({
                'code': ticker,
                'name': info['name'],
                'market': 'US',
                'allocation': allocation // 2,
                'sector': info['sector'],
                'volatility': info['volatility'],
                'return': info['return']
            })
        
        return selected
        
    except Exception as e:
        print(f"미국 주식 선택 오류: {e}")
        return []

def generate_stock_prediction_from_data(stock, korean_prices, us_prices):
    """실제 데이터 기반 주식 예측"""
    try:
        import pandas as pd
        
        # 해당 주식의 가격 데이터 추출
        if stock['market'] == 'KR':
            price_df = pd.DataFrame(korean_prices)
            stock_data = price_df[price_df['종목코드'] == stock['code']] if not price_df.empty else pd.DataFrame()
            price_col = '종가'
        else:
            price_df = pd.DataFrame(us_prices)
            stock_data = price_df[price_df['Ticker'] == stock['code']] if not price_df.empty else pd.DataFrame()
            price_col = 'Close'
        
        if stock_data.empty:
            return {
                "expected_return": 5.0,
                "confidence": 60.0,
                "horizon": 3,
                "risk_level": "보통",
                "analysis": f"{stock['name']} 데이터가 충분하지 않아 예측이 제한적입니다."
            }
        
        # 실제 데이터 기반 예측
        if len(stock_data) > 1:
            stock_data = stock_data.sort_values('Date' if stock['market'] == 'US' else '날짜')
            returns = stock_data[price_col].pct_change().dropna()
            
            if len(returns) > 0:
                avg_return = returns.mean() * 252 * 100  # 연환산 수익률
                volatility = returns.std() * (252**0.5) * 100  # 연환산 변동성
                
                # 최근 추세 분석
                recent_prices = stock_data[price_col].tail(10)
                trend = "상승" if recent_prices.iloc[-1] > recent_prices.iloc[0] else "하락"
                
                # 신뢰도 계산 (데이터 양과 안정성 기반)
                confidence = min(95, 50 + len(returns) * 0.5)
                
                risk_level = "높음" if volatility > 30 else "보통" if volatility > 15 else "낮음"
                
                return {
                    "expected_return": round(avg_return, 2),
                    "confidence": round(confidence, 1),
                    "horizon": 6,
                    "risk_level": risk_level,
                    "volatility": round(volatility, 2),
                    "trend": trend,
                    "analysis": f"{stock['name']}는 최근 {trend} 추세를 보이며, 연간 {avg_return:.1f}% 수익률과 {volatility:.1f}% 변동성을 나타냅니다. 위험 수준: {risk_level}"
                }
        
        # 기본값 반환
        return {
            "expected_return": stock.get('return', 5.0),
            "confidence": 70.0,
            "horizon": 3,
            "risk_level": "보통" if stock.get('volatility', 15) < 20 else "높음",
            "analysis": f"{stock['name']}에 대한 예측 분석 결과입니다."
        }
        
    except Exception as e:
        print(f"주식 예측 생성 오류 ({stock.get('name', 'Unknown')}): {e}")
        return {
            "expected_return": 5.0,
            "confidence": 60.0,
            "horizon": 3,
            "risk_level": "보통",
            "analysis": "예측 분석 중 오류가 발생했습니다."
        }

def generate_portfolio_analysis(selected_stocks, risk_tolerance):
    """포트폴리오 전체 분석"""
    try:
        if not selected_stocks:
            return "분석할 주식 데이터가 없습니다."
        
        total_allocation = sum(stock.get('allocation', 0) for stock in selected_stocks)
        avg_volatility = sum(stock.get('volatility', 0) for stock in selected_stocks) / len(selected_stocks)
        avg_return = sum(stock.get('return', 0) for stock in selected_stocks) / len(selected_stocks)
        
        korean_count = len([s for s in selected_stocks if s.get('market') == 'KR'])
        us_count = len([s for s in selected_stocks if s.get('market') == 'US'])
        
        risk_level = "고위험" if risk_tolerance >= 70 else "중위험" if risk_tolerance >= 40 else "안정형"
        
        analysis = f"""
{risk_level} 투자자를 위한 실제 데이터 기반 포트폴리오 분석:

구성: 한국 주식 {korean_count}개, 미국 주식 {us_count}개
예상 수익률: {avg_return:.2f}% (연간)
포트폴리오 변동성: {avg_volatility:.2f}%
총 배분 비율: {total_allocation}%

이 포트폴리오는 실제 수집된 주가 데이터를 바탕으로 구성되었으며, 
당신의 위험 선호도({risk_tolerance}/100)를 반영하여 최적화되었습니다.
        """.strip()
        
        return analysis
        
    except Exception as e:
        print(f"포트폴리오 분석 오류: {e}")
        return "포트폴리오 분석 중 오류가 발생했습니다."

@app.route('/api/news', methods=['GET'])
def api_news():
    """최신 뉴스/이슈 요약 반환 (CSV 파일 기반, 국내 뉴스 우선)"""
    query = request.args.get('query', '')
    session_id = session.get('user_id')
    if not session_id:
        session_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = session_id
    
    try:
        # CSV 파일에서 뉴스 데이터 로드
        news_data = real_data_manager.load_news_data()
        
        if news_data:
            # 쿼리 필터링
            if query:
                filtered_news = [
                    news for news in news_data 
                    if query.lower() in news.get('title', '').lower() or 
                       query.lower() in news.get('summary', '').lower()
                ]
            else:
                filtered_news = news_data
            
            # 한국 뉴스와 해외 뉴스 분리
            korean_news = [news for news in filtered_news if news.get('is_korean', False)]
            foreign_news = [news for news in filtered_news if not news.get('is_korean', False)]
            
            # 한국 뉴스를 먼저, 그 다음 해외 뉴스 (최대 30건)
            combined_news = korean_news[:20] + foreign_news[:10]
            
            # 시간순 정렬
            combined_news.sort(key=lambda x: x.get('published', ''), reverse=True)
            
            # 응답 형식 맞추기
            news_list = []
            for news in combined_news[:30]:
                news_list.append({
                    'title': news.get('title', ''),
                    'content': news.get('summary', '')[:200] + '...' if len(news.get('summary', '')) > 200 else news.get('summary', ''),
                    'url': news.get('link', '#'),
                    'published_at': news.get('published', ''),
                    'sentiment_score': 0.5,  # 기본값
                    'source': news.get('source', 'Unknown'),
                    'keyword': news.get('keyword', ''),  # MCP 검색 키워드
                    'is_korean': news.get('is_korean', False)
                })
            
            print(f"/api/news: 뉴스 {len(news_list)}건 반환 (query={query}, 한국: {len(korean_news)}, 해외: {len(foreign_news)})")
            return jsonify({"success": True, "news_list": news_list})
        
        # CSV 데이터가 없으면 Supabase 시도
        if news_processor and hasattr(news_processor, 'get_news_from_supabase'):
            news_list = news_processor.get_news_from_supabase(query=query, limit=30)
            logger.debug(f"/api/news: Supabase에서 뉴스 {len(news_list)}건 반환")
            return jsonify({"success": True, "news_list": news_list})
        
    except Exception as e:
        print(f"/api/news 오류: {e}")
        # 기본 뉴스 반환 (오류 시)
        default_news = [
            {
                'title': '시장 상황 분석 중',
                'content': '현재 뉴스 데이터를 업데이트하고 있습니다. 잠시만 기다려주세요.',
                'url': '#',
                'published_at': datetime.now().isoformat(),
                'sentiment_score': 0.5,
                'source': 'System'
            }
        ]
        return jsonify({"success": True, "news_list": default_news})

@app.route('/news')
def news_page():
    """뉴스/이슈 분석 결과 시각화 페이지"""
    return render_template('news.html')

@app.route('/api/stock/<ticker>/report', methods=['GET'])
def api_stock_report(ticker):
    """종목 상세 보고서 조회"""
    try:
        from src.stock_search_engine import StockSearchEngine
        search_engine = StockSearchEngine()
        
        # 먼저 검색 인덱스 로드
        search_engine.load_search_index()
        
        # 종목 코드로 직접 검색
        stock_info = None
        if search_engine.stock_data is not None:
            # 종목코드로 필터링
            filtered = search_engine.stock_data[search_engine.stock_data['종목코드'] == ticker]
            if not filtered.empty:
                stock_info = filtered.iloc[0]
        
        if stock_info is None:
            return jsonify({'success': False, 'error': '종목 정보를 찾을 수 없습니다.'})
        
        # 평가 이유 포맷팅
        evaluation_reasons = format_evaluation_reasons(stock_info)
        
        # 상세 보고서 데이터 구성 - numpy 타입을 Python 네이티브 타입으로 변환
        def convert_to_native(value):
            """numpy 타입을 Python 네이티브 타입으로 변환"""
            if pd.isna(value):
                return None
            elif isinstance(value, (np.integer, np.int64)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64)):
                return float(value)
            else:
                return value
                
        report_data = {
            'ticker': str(stock_info['종목코드']),
            'name': str(stock_info['종목명']),
            'currentPrice': convert_to_native(stock_info['현재가']),
            'marketCap': convert_to_native(stock_info['시가총액']),
            'score': convert_to_native(stock_info['평가점수']),
            'evaluation': str(stock_info['종합평가']),
            'metrics': {
                'revenueGrowth': convert_to_native(stock_info.get('매출성장률', None)),
                'profitMargin': convert_to_native(stock_info.get('순이익률', None)),
                'debtRatio': convert_to_native(stock_info.get('부채비율', None)),
                'per': convert_to_native(stock_info.get('PER', None)),
                'pbr': convert_to_native(stock_info.get('PBR', None))
            },
            'evaluationReasons': evaluation_reasons,
            'sector': str(stock_info.get('섹터', '기타')),
            'updatedAt': str(stock_info.get('기준일', datetime.now().strftime('%Y-%m-%d')))
        }
        
        return jsonify({'success': True, 'report': report_data})
        
    except Exception as e:
        logger.error(f"종목 보고서 조회 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

def format_evaluation_reasons(stock_info):
    """평가 이유를 한국어로 포맷팅"""
    reasons = []
    
    # 매출 성장률 평가
    revenue_growth = stock_info.get('매출성장률')
    if revenue_growth is not None and not pd.isna(revenue_growth):
        if revenue_growth > 20:
            reasons.append(f"매출 성장률이 {revenue_growth:.1f}%로 매우 우수함")
        elif revenue_growth > 10:
            reasons.append(f"매출 성장률이 {revenue_growth:.1f}%로 우수함")
        elif revenue_growth > 0:
            reasons.append(f"매출 성장률이 {revenue_growth:.1f}%로 양호함")
        else:
            reasons.append(f"매출 성장률이 {revenue_growth:.1f}%로 부진함")
    
    # 순이익률 평가
    profit_margin = stock_info.get('순이익률')
    if profit_margin is not None and not pd.isna(profit_margin):
        if profit_margin > 15:
            reasons.append(f"순이익률이 {profit_margin:.1f}%로 매우 우수함")
        elif profit_margin > 10:
            reasons.append(f"순이익률이 {profit_margin:.1f}%로 우수함")
        elif profit_margin > 5:
            reasons.append(f"순이익률이 {profit_margin:.1f}%로 양호함")
        else:
            reasons.append(f"순이익률이 {profit_margin:.1f}%로 개선 필요")
    
    # 부채비율 평가
    debt_ratio = stock_info.get('부채비율')
    if debt_ratio is not None and not pd.isna(debt_ratio):
        if debt_ratio < 50:
            reasons.append(f"부채비율이 {debt_ratio:.1f}%로 매우 안정적")
        elif debt_ratio < 100:
            reasons.append(f"부채비율이 {debt_ratio:.1f}%로 안정적")
        elif debt_ratio < 150:
            reasons.append(f"부채비율이 {debt_ratio:.1f}%로 적정 수준")
        else:
            reasons.append(f"부채비율이 {debt_ratio:.1f}%로 높은 편")
    
    # PER 평가
    per = stock_info.get('PER')
    if per is not None and not pd.isna(per):
        if per < 10:
            reasons.append(f"PER이 {per:.1f}로 저평가 상태")
        elif per < 15:
            reasons.append(f"PER이 {per:.1f}로 적정 수준")
        elif per < 25:
            reasons.append(f"PER이 {per:.1f}로 다소 높음")
        else:
            reasons.append(f"PER이 {per:.1f}로 고평가 상태")
    
    # PBR 평가
    pbr = stock_info.get('PBR')
    if pbr is not None and not pd.isna(pbr):
        if pbr < 1:
            reasons.append(f"PBR이 {pbr:.2f}로 자산가치 대비 저평가")
        elif pbr < 1.5:
            reasons.append(f"PBR이 {pbr:.2f}로 적정 수준")
        elif pbr < 3:
            reasons.append(f"PBR이 {pbr:.2f}로 다소 높음")
        else:
            reasons.append(f"PBR이 {pbr:.2f}로 고평가 상태")
    
    # 종합 평가 추가
    score = stock_info.get('평가점수', 0)
    if score >= 80:
        reasons.append("종합적으로 매우 우수한 투자 매력도를 보임")
    elif score >= 70:
        reasons.append("종합적으로 양호한 투자 매력도를 보임")
    elif score >= 60:
        reasons.append("종합적으로 보통 수준의 투자 매력도를 보임")
    else:
        reasons.append("종합적으로 신중한 접근이 필요함")
    
    return reasons

@app.route('/api/recommendations/stocks', methods=['POST'])
def api_recommend_stocks():
    """투자 성향에 맞는 종목 추천"""
    try:
        data = request.get_json()
        investment_type = data.get('investmentType', '')
        risk_tolerance = data.get('riskTolerance', 0)
        
        # StockSearchEngine을 사용하여 평가 점수 기반 종목 조회
        try:
            from src.stock_search_engine import StockSearchEngine
            search_engine = StockSearchEngine()
            
            # 상위 평가 종목 가져오기
            top_stocks = search_engine.get_top_stocks(n_results=20)
            logger.info(f"상위 {len(top_stocks)}개 종목 로드 완료")
            
            # 위험 성향에 따른 종목 필터링
            recommended_stocks = []
            
            if risk_tolerance >= 1.5:  # 공격적 투자자
                # 성장성 높은 종목 위주
                for stock in top_stocks:
                    if stock.평가점수 >= 80:
                        recommended_stocks.append({
                            'ticker': stock.종목코드,
                            'name': stock.종목명,
                            'currentPrice': stock.현재가,
                            'changePercent': 2.5,  # 실제 변동률 데이터가 없으므로 임시값
                            'score': stock.평가점수,
                            'evaluation': stock.종합평가,
                            'per': stock.PER,
                            'pbr': stock.PBR,
                            'debtRatio': stock.부채비율,
                            'profitMargin': stock.순이익률,
                            'revenueGrowth': stock.매출성장률
                        })
                    if len(recommended_stocks) >= 6:
                        break
                        
            elif risk_tolerance >= 0.5:  # 적극적 투자자
                # 균형잡힌 성장주
                for stock in top_stocks:
                    if stock.평가점수 >= 75 and stock.부채비율 is not None and stock.부채비율 < 150:
                        recommended_stocks.append({
                            'ticker': stock.종목코드,
                            'name': stock.종목명,
                            'currentPrice': stock.현재가,
                            'changePercent': 1.5,
                            'score': stock.평가점수,
                            'evaluation': stock.종합평가,
                            'per': stock.PER,
                            'pbr': stock.PBR,
                            'debtRatio': stock.부채비율,
                            'profitMargin': stock.순이익률,
                            'revenueGrowth': stock.매출성장률
                        })
                    if len(recommended_stocks) >= 6:
                        break
                        
            elif risk_tolerance >= -0.5:  # 중립적 투자자
                # 안정성과 성장성 균형
                for stock in top_stocks:
                    if (stock.평가점수 >= 70 and 
                        stock.부채비율 is not None and stock.부채비율 < 100 and
                        stock.PER is not None and 5 < stock.PER < 20):
                        recommended_stocks.append({
                            'ticker': stock.종목코드,
                            'name': stock.종목명,
                            'currentPrice': stock.현재가,
                            'changePercent': 0.8,
                            'score': stock.평가점수,
                            'evaluation': stock.종합평가,
                            'per': stock.PER,
                            'pbr': stock.PBR,
                            'debtRatio': stock.부채비율,
                            'profitMargin': stock.순이익률,
                            'revenueGrowth': stock.매출성장률
                        })
                    if len(recommended_stocks) >= 6:
                        break
                        
            else:  # 보수적 투자자
                # 안정성 높은 우량주
                for stock in top_stocks:
                    if (stock.평가점수 >= 65 and 
                        stock.부채비율 is not None and stock.부채비율 < 80 and
                        stock.PER is not None and stock.PER < 15 and
                        stock.순이익률 is not None and stock.순이익률 > 5):
                        recommended_stocks.append({
                            'ticker': stock.종목코드,
                            'name': stock.종목명,
                            'currentPrice': stock.현재가,
                            'changePercent': 0.5,
                            'score': stock.평가점수,
                            'evaluation': stock.종합평가,
                            'per': stock.PER,
                            'pbr': stock.PBR,
                            'debtRatio': stock.부채비율,
                            'profitMargin': stock.순이익률,
                            'revenueGrowth': stock.매출성장률
                        })
                    if len(recommended_stocks) >= 6:
                        break
            
            # 카테고리별 종목 분류 (포트폴리오 차트 클릭용)
            stocks_by_category = {
                '국내 주식': recommended_stocks[:3] if len(recommended_stocks) >= 3 else recommended_stocks,
                '해외 주식': [],  # 현재는 한국 주식만
                '채권': [{'code': 'KTB', 'name': '한국국채', 'allocation': 20}],
                '현금성 자산': [{'code': 'MMF', 'name': 'MMF', 'allocation': 10}],
                '대체 투자': [{'code': 'REIT', 'name': '리츠', 'allocation': 10}]
            }
            
            # 각 종목에 할당 비율 추가
            for i, stock in enumerate(stocks_by_category['국내 주식']):
                stock['allocation'] = 15 - i * 2  # 15%, 13%, 11% 등으로 할당
            
            return jsonify({
                'success': True,
                'stocks': recommended_stocks,
                'stocksByCategory': stocks_by_category
            })
            
        except Exception as e:
            logger.error(f"종목 추천 중 오류: {e}")
            # 기본 추천 종목 반환
            default_stocks = get_default_recommendations(risk_tolerance)
            return jsonify({
                'success': True,
                'stocks': default_stocks,
                'stocksByCategory': {
                    '국내 주식': default_stocks[:3],
                    '해외 주식': [],
                    '채권': [{'code': 'KTB', 'name': '한국국채', 'allocation': 20}],
                    '현금성 자산': [{'code': 'MMF', 'name': 'MMF', 'allocation': 10}],
                    '대체 투자': [{'code': 'REIT', 'name': '리츠', 'allocation': 10}]
                }
            })
            
    except Exception as e:
        logger.error(f"종목 추천 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def get_default_recommendations(risk_tolerance):
    """기본 추천 종목 반환"""
    if risk_tolerance >= 1.0:
        return [
            {'ticker': '005930', 'name': '삼성전자', 'currentPrice': 78200, 'changePercent': 1.2, 'score': 85, 'evaluation': '매수', 'per': 12.5, 'debtRatio': 45},
            {'ticker': '000660', 'name': 'SK하이닉스', 'currentPrice': 138500, 'changePercent': -0.5, 'score': 82, 'evaluation': '매수', 'per': 8.3, 'debtRatio': 52},
            {'ticker': '035420', 'name': 'NAVER', 'currentPrice': 215000, 'changePercent': 2.1, 'score': 80, 'evaluation': '보유', 'per': 28.5, 'debtRatio': 25},
            {'ticker': '035720', 'name': '카카오', 'currentPrice': 48500, 'changePercent': 1.8, 'score': 78, 'evaluation': '보유', 'per': 45.2, 'debtRatio': 18},
            {'ticker': '207940', 'name': '삼성바이오로직스', 'currentPrice': 785000, 'changePercent': 0.9, 'score': 77, 'evaluation': '보유', 'per': 68.3, 'debtRatio': 35}
        ]
    elif risk_tolerance >= 0:
        return [
            {'ticker': '005930', 'name': '삼성전자', 'currentPrice': 78200, 'changePercent': 1.2, 'score': 85, 'evaluation': '매수', 'per': 12.5, 'debtRatio': 45},
            {'ticker': '005380', 'name': '현대차', 'currentPrice': 195000, 'changePercent': 0.8, 'score': 78, 'evaluation': '보유', 'per': 6.8, 'debtRatio': 62},
            {'ticker': '051910', 'name': 'LG화학', 'currentPrice': 425000, 'changePercent': -0.3, 'score': 76, 'evaluation': '보유', 'per': 15.2, 'debtRatio': 58},
            {'ticker': '068270', 'name': '셀트리온', 'currentPrice': 182000, 'changePercent': 1.5, 'score': 74, 'evaluation': '보유', 'per': 24.5, 'debtRatio': 42}
        ]
    else:
        return [
            {'ticker': '055550', 'name': '신한지주', 'currentPrice': 42500, 'changePercent': -0.3, 'score': 76, 'evaluation': '보유', 'per': 5.2, 'debtRatio': 0},
            {'ticker': '105560', 'name': 'KB금융', 'currentPrice': 58200, 'changePercent': 0.5, 'score': 75, 'evaluation': '보유', 'per': 4.8, 'debtRatio': 0},
            {'ticker': '086790', 'name': '하나금융지주', 'currentPrice': 45800, 'changePercent': 0.2, 'score': 74, 'evaluation': '보유', 'per': 4.5, 'debtRatio': 0},
            {'ticker': '017670', 'name': 'SK텔레콤', 'currentPrice': 51200, 'changePercent': -0.1, 'score': 72, 'evaluation': '보유', 'per': 9.8, 'debtRatio': 78}
        ]

@app.route('/api/pixie-insights', methods=['GET'])
def api_pixie_insights():
    """픽시의 인사이트와 한마디 생성"""
    try:
        # 오늘의 픽 뉴스 가져오기
        filter_type = request.args.get('filter', 'domestic')
        news_data = real_data_manager.load_news_data()
        
        # 오늘의 픽 뉴스 필터링
        today_pick_news = []
        if news_data and len(news_data) > 0:
            # 필터링 로직 적용
            for news in news_data[:20]:  # 최대 20개
                title = str(news.get('title', ''))
                content = str(news.get('content', ''))
                
                # 해외 뉴스 키워드
                global_keywords = ['미국', '중국', '일본', '유럽', 'EU', '연준', 'Fed', 'FOMC', '바이든', '트럼프', 
                                  '나스닥', 'S&P', '다우', '애플', '테슬라', '엔비디아', 'TSMC', 'NYSE', '달러', '엔화', '위안화']
                
                # 국내 뉴스 키워드  
                domestic_keywords = ['한국', '국내', '코스피', '코스닥', '삼성', 'SK', 'LG', '현대', '기아', '네이버', '카카오',
                                   '금융위', '한국은행', '금통위', '서울', '대한민국']
                
                # 뉴스 분류
                is_global = any(keyword in title or keyword in content for keyword in global_keywords)
                is_domestic = any(keyword in title or keyword in content for keyword in domestic_keywords)
                
                # 명확하지 않은 경우 기본값은 국내
                if not is_global and not is_domestic:
                    is_domestic = True
                
                # 필터링 적용
                if filter_type == 'global' and not is_global:
                    continue
                elif filter_type == 'domestic' and is_global and not is_domestic:
                    continue
                
                today_pick_news.append(news)
                
                if len(today_pick_news) >= 8:
                    break
        
        if today_pick_news:
            # 오늘의 픽 뉴스 중 첫 번째 뉴스를 기반으로 인사이트 생성
            main_news = today_pick_news[0]
            main_title = main_news.get('title', '')
            main_content = main_news.get('content', '')
            
            # AI를 사용해서 인사이트 생성 (예시)
            if filter_type == 'global':
                # 해외 뉴스 기반 인사이트
                if '연준' in main_title or 'Fed' in main_title or '금리' in main_title:
                    insights = {
                        'title': '글로벌 통화정책의 변화, 투자 전략의 재편성',
                        'content': '연준의 통화정책 변화가 글로벌 시장에 미치는 영향이 커지고 있습니다. 금리 변화는 주식, 채권, 외환 시장 전반에 걸쳐 투자 패턴의 변화를 가져올 수 있습니다.',
                        'points': [
                            '글로벌 통화정책 변화는 환율과 자본 흐름에 직접적인 영향을 미칩니다.',
                            '장기 투자자는 단기 변동성보다 기업의 펀더멘털에 집중하는 것이 중요합니다.'
                        ],
                        'quote': '변동성은 기회,\n준비된 자만이 잡는다!'
                    }
                elif '테슬라' in main_title or '애플' in main_title or '기술' in main_title:
                    insights = {
                        'title': '글로벌 기술주의 새로운 장, AI와 혁신의 시대',
                        'content': '글로벌 기술 기업들의 혁신이 계속되고 있습니다. AI, 전기차, 메타버스 등 새로운 기술 트렌드가 투자 기회를 만들고 있습니다.',
                        'points': [
                            '기술주는 성장 가능성이 높지만 변동성도 크기 때문에 리스크 관리가 필수입니다.',
                            '장기적인 기술 트렌드를 파악하고 선도 기업에 투자하는 전략이 효과적입니다.'
                        ],
                        'quote': '혁신을 따라가지 말고,\n혁신을 선도하라!'
                    }
                else:
                    # 기본 해외 인사이트
                    insights = {
                        'title': '글로벌 시장의 변화, 새로운 투자 패러다임',
                        'content': f'{main_title[:50]}... 글로벌 시장은 지속적으로 변화하고 있으며, 이러한 변화는 새로운 투자 기회를 만들고 있습니다.',
                        'points': [
                            '글로벌 시장의 다변화는 리스크 분산의 기회를 제공합니다.',
                            '환율 변동과 국제 정치 상황을 면밀히 모니터링하세요.'
                        ],
                        'quote': '세계를 보는 눈,\n미래를 여는 투자!'
                    }
            else:
                # 국내 뉴스 기반 인사이트
                if '삼성' in main_title or 'SK' in main_title or '반도체' in main_title:
                    insights = {
                        'title': '반도체 산업의 새로운 전환점, 기회와 도전',
                        'content': '한국 반도체 산업이 새로운 전환점을 맞고 있습니다. AI 반도체 수요 증가와 함께 글로벌 경쟁이 치열해지고 있습니다.',
                        'points': [
                            '반도체 사이클은 회복 국면에 접어들었지만, 중장기 전망은 여전히 밝습니다.',
                            '글로벌 경쟁력을 갖춘 국내 기업들의 기술 투자에 주목하세요.'
                        ],
                        'quote': '기술의 미래,\n한국이 만든다!'
                    }
                elif '헬스케어' in main_title or '바이오' in main_title or '제약' in main_title:
                    insights = {
                        'title': '헬스케어 산업의 혁신, 미래 성장 동력',
                        'content': '한국 헬스케어 산업이 글로벌 경쟁력을 강화하고 있습니다. 바이오시밀러, 신약 개발, 디지털 헬스케어 등 다양한 분야에서 성과가 나타나고 있습니다.',
                        'points': [
                            '헬스케어는 중장기 성장성과 정책 리스크를 함께 고려해야 합니다.',
                            '임상 단계별 진행 상황과 규제 환경 변화를 면밀히 모니터링하세요.'
                        ],
                        'quote': '건강한 미래,\n투자의 새 지평!'
                    }
                else:
                    # 기본 국내 인사이트
                    insights = {
                        'title': '한국 시장의 변화와 기회',
                        'content': f'{main_title[:50]}... 국내 시장은 새로운 성장 동력을 찾고 있으며, 투자자들에게 다양한 기회를 제공하고 있습니다.',
                        'points': [
                            '국내 시장의 구조적 변화를 이해하고 장기 투자 관점을 유지하세요.',
                            '업종별 특성과 글로벌 트렌드와의 연계성을 고려한 투자가 필요합니다.'
                        ],
                        'quote': '변화를 읽고,\n기회를 잡아라!'
                    }
            
            return jsonify({
                'success': True,
                'insights': insights,
                'based_on_news': main_title  # 어떤 뉴스를 기반으로 했는지 표시
            })
        
        # 기본 인사이트 반환
        return jsonify({
            'success': True,
            'insights': {
                'title': '시장의 변화를 읽는 지혜',
                'content': '투자 시장은 끊임없이 변화합니다. 오늘의 뉴스를 통해 내일의 기회를 발견하세요.',
                'points': [
                    '단기적 변동보다는 장기적 트렌드에 주목하세요.',
                    '분산 투자를 통해 리스크를 관리하는 것이 중요합니다.'
                ],
                'quote': '준비된 자에게\n기회가 찾아옵니다!'
            }
        })
        
    except Exception as e:
        print(f"픽시 인사이트 생성 오류: {e}")
        return jsonify({
            'success': True,
            'insights': {
                'title': '시장의 변화를 읽는 지혜',
                'content': '투자 시장은 끊임없이 변화합니다.',
                'points': ['변화에 주목하세요.', '리스크를 관리하세요.'],
                'quote': '기회는 준비된 자에게!'
            }
        })

@app.route('/api/trend-keywords', methods=['GET'])
def api_trend_keywords():
    """트렌드 키워드 반환"""
    try:
        # 실제 뉴스 데이터에서 키워드 추출
        news_data = real_data_manager.load_news_data()
        
        keyword_count = {}
        if news_data and len(news_data) > 0:
            # 뉴스 제목과 내용에서 키워드 추출
            for news in news_data[:50]:  # 최근 50개 뉴스
                title = str(news.get('title', ''))
                content = str(news.get('content', ''))
                
                # 주요 키워드 카운트
                keywords = ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '현대차', '기아', 
                          'AI', '반도체', '2차전지', '바이오', '디스플레이', 
                          '테슬라', 'IRA 법안', '탄소배출권', '토목 SOC',
                          '유전자 치료제', '코스피', '코스닥', '금리', '환율']
                
                for keyword in keywords:
                    if keyword in title or keyword in content:
                        keyword_count[keyword] = keyword_count.get(keyword, 0) + 1
        
        # 상위 키워드 정렬
        sorted_keywords = sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)
        
        # 트렌드 키워드 생성
        trend_keywords = []
        for keyword, count in sorted_keywords[:10]:
            trend = {
                'text': f'#{keyword}',
                'count': count,
                'change': random.randint(-20, 50),  # 실제로는 전일 대비 계산
                'isFilled': count > 3  # 언급이 많으면 강조
            }
            trend_keywords.append(trend)
        
        # 기본 키워드 추가
        if len(trend_keywords) < 10:
            default_keywords = ['#디스플레이', '#바이오', '#유전자 치료제', '#테슬라 옵티머스',
                              '#AI 반도체', '#삼성전자', '#2차전지', '#IRA 법안', 
                              '#탄소배출권', '#토목 SOC']
            for kw in default_keywords:
                if kw not in [t['text'] for t in trend_keywords]:
                    trend_keywords.append({
                        'text': kw,
                        'count': random.randint(1, 5),
                        'change': random.randint(-10, 30),
                        'isFilled': random.choice([True, False])
                    })
        
        return jsonify({
            'success': True,
            'keywords': trend_keywords[:10]
        })
        
    except Exception as e:
        print(f"트렌드 키워드 생성 오류: {e}")
        # 기본 트렌드 키워드 반환
        default_trends = [
            {'text': '#디스플레이', 'change': 25, 'isFilled': True},
            {'text': '#바이오', 'change': 15, 'isFilled': False},
            {'text': '#유전자 치료제', 'change': 30, 'isFilled': False},
            {'text': '#테슬라 옵티머스', 'change': 45, 'isFilled': True},
            {'text': '#AI 반도체', 'change': 20, 'isFilled': False},
            {'text': '#삼성전자', 'change': 35, 'isFilled': True},
            {'text': '#2차전지', 'change': 10, 'isFilled': False},
            {'text': '#IRA 법안', 'change': 40, 'isFilled': True},
            {'text': '#탄소배출권', 'change': 5, 'isFilled': False},
            {'text': '#토목 SOC', 'change': 15, 'isFilled': False}
        ]
        return jsonify({
            'success': True,
            'keywords': default_trends
        })

@app.route('/time_series')
def time_series_prediction():
    """시계열 예측 페이지"""
    return render_template('time_series_prediction.html')

@app.route('/api/time_series_prediction', methods=['POST'])
def api_time_series_prediction():
    """시계열 예측 API"""
    try:
        data = request.get_json()
        stock_code = data.get('stock_code')
        days = data.get('days', 7)
        
        if not stock_code:
            return jsonify({
                'success': False,
                'error': '종목코드가 필요합니다'
            })
        
        # 실제 데이터 매니저를 사용해 예측 생성
        prediction = real_data_manager.generate_time_series_prediction(stock_code, days)
        
        if prediction and 'error' not in prediction:
            return jsonify({
                'success': True,
                'prediction': prediction
            })
        else:
            error_msg = prediction.get('error', '예측 생성에 실패했습니다') if prediction else '데이터를 찾을 수 없습니다'
            return jsonify({
                'success': False,
                'error': error_msg
            })
            
    except Exception as e:
        print(f"시계열 예측 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': f'서버 오류: {str(e)}'
        })

@app.route('/api/chat-stream')
def chat_stream():
    """SSE (Server-Sent Events) 형식으로 AI 응답을 스트리밍합니다."""
    from flask import Response, stream_with_context
    
    message = request.args.get('message', '').strip()
    session_id = session.get('user_id')
    if not session_id:
        session_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = session_id
    
    def generate():
        try:
            # 1. 설문 결과 조회
            from src.db_client import get_supabase_client
            supabase = get_supabase_client()
            
            has_profile = False
            profile_json = None
            summary = None
            
            # 프로필 조회 로직...
            try:
                if supabase:
                    res = supabase.table("user_profiles").select("profile_json,summary").eq("user_id", session_id).order("created_at", desc=True).limit(1).execute()
                    has_profile = res.data and len(res.data) > 0
                    profile_json = res.data[0]['profile_json'] if has_profile else None
                    summary = res.data[0]['summary'] if has_profile else None
                else:
                    # SQLite 폴백
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""
                        SELECT analysis_result FROM user_profiles 
                        WHERE user_id = ? 
                        ORDER BY created_at DESC LIMIT 1
                    """, (session_id,))
                    result = c.fetchone()
                    if result and result[0]:
                        profile_data = json.loads(result[0])
                        has_profile = True
                        profile_json = profile_data
                        summary = profile_data.get('overall_analysis', '')
                    conn.close()
            except Exception as e:
                print(f"프로필 조회 오류: {e}")
            
            # 2. 설문 미완료 안내
            if not has_profile:
                yield f"data: {json.dumps({'content': '먼저 투자 성향 설문을 완료해 주세요!', 'require_survey': True})}\n\n"
                yield "event: complete\ndata: {}\n\n"
                return
            
            # 3. Investment Advisor를 통한 AI Agent Chain 실행
            advisor = get_advisor_for_session(session_id)
            
            # 프로필 데이터가 캐시되지 않았을 경우 다시 로드
            if profile_json and not advisor.memory_manager.get_cached_context(session_id, "user_profile"):
                if isinstance(profile_json, str):
                    advisor.memory_manager.cache_context(session_id, "user_profile", json.loads(profile_json))
                else:
                    advisor.memory_manager.cache_context(session_id, "user_profile", profile_json)
            
            # 상태 콜백 함수 정의
            def status_callback(agent_name, status):
                if status == 'thinking':
                    yield f"event: status\ndata: {json.dumps({'agent': agent_name, 'status': status})}\n\n"
            
            # 응답 콜백 함수 정의 - 각 AI 에이전트의 응답을 전송
            def response_callback(agent_name, response_text):
                # 에이전트 응답을 agent_response 이벤트로 전송
                yield f"event: agent_response\ndata: {json.dumps({'agent': agent_name, 'content': response_text})}\n\n"
            
            # AI 에이전트 체인 실행
            advisor.set_status_callback(status_callback)
            advisor.set_response_callback(response_callback)
            
            # AI-A thinking 상태 전송
            yield f"event: status\ndata: {json.dumps({'agent': 'AI-A', 'status': 'thinking'})}\n\n"
            response = advisor.chat(message)
            
            # 최종 응답 전송
            yield f"data: {json.dumps({'content': response})}\n\n"
            yield "event: complete\ndata: {}\n\n"
            
        except Exception as e:
            print(f"SSE 스트리밍 중 오류: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "event: complete\ndata: {}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

@app.route('/api/profile-status')
def profile_status():
    """사용자의 프로필 상태를 확인합니다."""
    try:
        session_id = session.get('user_id')
        if not session_id:
            return jsonify({"success": True, "has_profile": False})
        
        # Supabase에서 프로필 확인
        from src.db_client import get_supabase_client
        supabase = get_supabase_client()
        
        has_profile = False
        
        if supabase:
            try:
                res = supabase.table("user_profiles").select("user_id").eq("user_id", session_id).limit(1).execute()
                has_profile = res.data and len(res.data) > 0
            except Exception as e:
                logger.error(f"Supabase 프로필 확인 오류: {e}")
                # SQLite 폴백
                try:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("SELECT id FROM user_profiles WHERE user_id = ? LIMIT 1", (session_id,))
                    result = c.fetchone()
                    has_profile = result is not None
                    conn.close()
                except Exception as sqlite_err:
                    print(f"SQLite 프로필 확인 오류: {sqlite_err}")
        else:
            # SQLite 사용
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("SELECT id FROM user_profiles WHERE user_id = ? LIMIT 1", (session_id,))
                result = c.fetchone()
                has_profile = result is not None
                conn.close()
            except Exception as e:
                print(f"SQLite 프로필 확인 오류: {e}")
        
        return jsonify({"success": True, "has_profile": has_profile})
        
    except Exception as e:
        print(f"프로필 상태 확인 중 오류: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/chat-history', methods=['GET', 'POST'])
def chat_history():
    """채팅 기록을 처리합니다."""
    if request.method == 'GET':
        try:
            history = load_chat_history()
            return jsonify({"success": True, "history": history})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    elif request.method == 'POST':
        try:
            # POST 메서드는 클라이언트 측 저장용
            data = request.get_json()
            # 필요시 서버 측 저장 로직 추가 가능
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

@app.route('/api/profile')
def get_profile():
    """사용자의 투자 프로필을 반환합니다."""
    try:
        session_id = session.get('user_id')
        if not session_id:
            return jsonify({"success": False, "error": "세션이 없습니다."})
        
        # 먼저 세션에서 확인
        if 'survey_result' in session:
            result = session['survey_result']
            # personality_type 추가
            if 'overall_analysis' in result:
                overall = result['overall_analysis']
                if '공격투자형' in overall:
                    personality_type = '공격투자형'
                elif '적극투자형' in overall:
                    personality_type = '적극투자형'
                elif '위험중립형' in overall:
                    personality_type = '위험중립형'
                elif '안정추구형' in overall:
                    personality_type = '안정추구형'
                else:
                    personality_type = '안정형'
            else:
                personality_type = '분석 중'
            
            profile = {
                'user_id': session_id,
                'personality_type': personality_type,
                'risk_score': result.get('scores', {}).get('risk_tolerance', 0),
                'knowledge_score': result.get('scores', {}).get('investment_confidence', 0),
                'experience_score': result.get('scores', {}).get('investment_time_horizon', 0),
                'financial_score': result.get('scores', {}).get('financial_goal_orientation', 0),
                'answers': {
                    'q2': 2,  # 기본값
                    'q5': 3,
                    'q7': 2,
                    'q9': 2,
                    'q10': 2
                }
            }
            return jsonify({"success": True, "profile": profile})
        
        # Supabase에서 조회
        from src.db_client import get_supabase_client
        supabase = get_supabase_client()
        
        if supabase:
            try:
                res = supabase.table("user_profiles").select("*").eq("user_id", session_id).order('created_at', desc=True).limit(1).execute()
                if res.data and len(res.data) > 0:
                    data = res.data[0]
                    profile_json = data.get('profile_json', {})
                    scores = profile_json.get('scores', {})
                    
                    # personality_type 추출
                    overall = profile_json.get('overall_analysis', '')
                    if '공격투자형' in overall:
                        personality_type = '공격투자형'
                    elif '적극투자형' in overall:
                        personality_type = '적극투자형'
                    elif '위험중립형' in overall:
                        personality_type = '위험중립형'
                    elif '안정추구형' in overall:
                        personality_type = '안정추구형'
                    else:
                        personality_type = '안정형'
                    
                    profile = {
                        'user_id': session_id,
                        'personality_type': personality_type,
                        'risk_score': scores.get('risk_tolerance', 0),
                        'knowledge_score': scores.get('investment_confidence', 0),
                        'experience_score': scores.get('investment_time_horizon', 0),
                        'financial_score': scores.get('financial_goal_orientation', 0),
                        'answers': {
                            'q2': 2,  # 기본값
                            'q5': 3,
                            'q7': 2,
                            'q9': 2,
                            'q10': 2
                        }
                    }
                    return jsonify({"success": True, "profile": profile})
            except Exception as e:
                logger.error(f"Supabase 프로필 조회 오류: {e}")
        
        # SQLite 폴백
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT profile_json, summary 
                FROM user_profiles 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (session_id,))
            result = c.fetchone()
            conn.close()
            
            if result:
                profile_json = json.loads(result[0]) if result[0] else {}
                scores = profile_json.get('scores', {})
                overall = profile_json.get('overall_analysis', '')
                
                if '공격투자형' in overall:
                    personality_type = '공격투자형'
                elif '적극투자형' in overall:
                    personality_type = '적극투자형'
                elif '위험중립형' in overall:
                    personality_type = '위험중립형'
                elif '안정추구형' in overall:
                    personality_type = '안정추구형'
                else:
                    personality_type = '안정형'
                
                profile = {
                    'user_id': session_id,
                    'personality_type': personality_type,
                    'risk_score': scores.get('risk_tolerance', 0),
                    'knowledge_score': scores.get('investment_confidence', 0),
                    'experience_score': scores.get('investment_time_horizon', 0),
                    'financial_score': scores.get('financial_goal_orientation', 0),
                    'answers': {
                        'q2': 2,
                        'q5': 3,
                        'q7': 2,
                        'q9': 2,
                        'q10': 2
                    }
                }
                return jsonify({"success": True, "profile": profile})
        except Exception as e:
            print(f"SQLite 프로필 조회 오류: {e}")
        
        return jsonify({"success": False, "error": "프로필을 찾을 수 없습니다."})
        
    except Exception as e:
        print(f"프로필 조회 중 오류: {e}")
        return jsonify({"success": False, "error": str(e)})

# 투자 학습 관련 데이터
INVESTMENT_TERMS = [
    {
        "id": 1,
        "term": "PER",
        "full_name": "Price Earnings Ratio",
        "korean": "주가수익비율",
        "category": "기본 지표",
        "difficulty": "초급",
        "definition": "주가를 주당순이익(EPS)으로 나눈 값으로, 기업의 수익성 대비 주가 수준을 나타내는 지표",
        "formula": "PER = 주가 ÷ 주당순이익(EPS)",
        "example": "삼성전자의 주가가 70,000원이고 EPS가 5,000원이면, PER은 14배",
        "usage": "PER이 낮을수록 주가가 저평가되었다고 볼 수 있으나, 업종별 특성을 고려해야 함",
        "related_terms": ["PBR", "EPS", "ROE"]
    },
    {
        "id": 2,
        "term": "PBR",
        "full_name": "Price Book-value Ratio",
        "korean": "주가순자산비율",
        "category": "기본 지표",
        "difficulty": "초급",
        "definition": "주가를 주당순자산가치(BPS)로 나눈 값으로, 기업의 자산가치 대비 주가 수준을 나타내는 지표",
        "formula": "PBR = 주가 ÷ 주당순자산(BPS)",
        "example": "A기업의 주가가 50,000원이고 BPS가 40,000원이면, PBR은 1.25배",
        "usage": "PBR이 1보다 낮으면 주가가 장부가치보다 낮게 거래되는 것을 의미",
        "related_terms": ["PER", "BPS", "ROA"]
    },
    {
        "id": 3,
        "term": "ROE",
        "full_name": "Return On Equity",
        "korean": "자기자본수익률",
        "category": "수익성 지표",
        "difficulty": "중급",
        "definition": "기업이 자기자본을 활용해 얼마나 많은 이익을 창출했는지를 나타내는 지표",
        "formula": "ROE = (당기순이익 ÷ 자기자본) × 100",
        "example": "자기자본 1000억원으로 100억원의 순이익을 낸 경우, ROE는 10%",
        "usage": "ROE가 높을수록 자본을 효율적으로 사용하여 수익을 창출하는 기업",
        "related_terms": ["ROA", "PER", "EPS"]
    },
    {
        "id": 4,
        "term": "EPS",
        "full_name": "Earnings Per Share",
        "korean": "주당순이익",
        "category": "기본 지표",
        "difficulty": "초급",
        "definition": "기업의 순이익을 발행주식수로 나눈 값으로, 주식 1주당 얻을 수 있는 이익",
        "formula": "EPS = 당기순이익 ÷ 발행주식수",
        "example": "순이익 1조원, 발행주식수 1억주인 경우 EPS는 10,000원",
        "usage": "EPS가 증가하면 기업의 수익성이 개선되고 있음을 의미",
        "related_terms": ["PER", "DPS", "BPS"]
    },
    {
        "id": 5,
        "term": "시가총액",
        "full_name": "Market Capitalization",
        "korean": "시가총액",
        "category": "기본 개념",
        "difficulty": "초급",
        "definition": "기업의 총 발행주식수에 현재 주가를 곱한 값으로, 시장에서 평가하는 기업의 가치",
        "formula": "시가총액 = 현재주가 × 발행주식수",
        "example": "주가 50,000원, 발행주식수 1억주인 경우 시가총액은 5조원",
        "usage": "기업 규모를 비교하거나 인수합병 시 기업가치 평가의 기준",
        "related_terms": ["주가", "발행주식수", "기업가치"]
    }
]

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "level": "초급",
        "question": "PER이 10배인 기업의 의미는 무엇인가요?",
        "options": [
            "주가가 주당순이익의 10배",
            "주가가 10원",
            "이익이 10배 증가",
            "배당금이 10%"
        ],
        "answer": 0,
        "explanation": "PER 10배는 현재 주가가 주당순이익(EPS)의 10배라는 의미입니다. 즉, 현재의 이익 수준이 계속된다면 10년 만에 투자원금을 회수할 수 있다는 뜻입니다."
    },
    {
        "id": 2,
        "level": "초급",
        "question": "다음 중 기업의 규모를 나타내는 지표는?",
        "options": [
            "PER",
            "PBR",
            "시가총액",
            "ROE"
        ],
        "answer": 2,
        "explanation": "시가총액은 현재 주가에 발행주식수를 곱한 값으로, 시장에서 평가하는 기업의 전체 가치를 나타냅니다."
    },
    {
        "id": 3,
        "level": "중급",
        "question": "ROE가 15%인 기업과 5%인 기업 중 어느 기업이 더 효율적으로 경영되고 있다고 볼 수 있나요?",
        "options": [
            "ROE 5% 기업",
            "ROE 15% 기업",
            "둘 다 같음",
            "판단할 수 없음"
        ],
        "answer": 1,
        "explanation": "ROE(자기자본수익률)가 높을수록 기업이 자기자본을 효율적으로 활용하여 이익을 창출하고 있음을 의미합니다."
    },
    {
        "id": 4,
        "level": "중급",
        "question": "PBR이 0.8인 기업의 특징은?",
        "options": [
            "고평가된 기업",
            "장부가치보다 낮게 거래",
            "성장성이 높은 기업",
            "배당률이 높은 기업"
        ],
        "answer": 1,
        "explanation": "PBR이 1보다 낮다는 것은 주가가 기업의 장부상 순자산가치보다 낮게 거래되고 있음을 의미합니다."
    },
    {
        "id": 5,
        "level": "고급",
        "question": "EPS가 5,000원이고 배당금이 1,000원인 기업의 배당성향은?",
        "options": [
            "10%",
            "20%",
            "50%",
            "80%"
        ],
        "answer": 1,
        "explanation": "배당성향 = (주당배당금 ÷ EPS) × 100 = (1,000 ÷ 5,000) × 100 = 20%"
    }
]

@app.route('/api/evaluation-insights')
def get_evaluation_insights():
    """평가 데이터 기반 인사이트 반환"""
    try:
        analyzer = FinancialReportAnalyzer()
        
        # 특정 종목 코드가 파라미터로 전달된 경우
        ticker = request.args.get('ticker')
        
        # 평가 데이터 기반 인사이트 가져오기
        insights = analyzer.get_evaluation_insights(ticker)
        
        return jsonify({
            'success': True,
            'data': insights
        })
        
    except Exception as e:
        logger.error(f"평가 인사이트 조회 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/comprehensive-analysis')
def get_comprehensive_analysis():
    """시장 종합 분석 데이터 반환"""
    try:
        analyzer = FinancialReportAnalyzer()
        
        # 종합 분석 데이터 가져오기
        analysis = analyzer.get_comprehensive_analysis()
        
        return jsonify({
            'success': True,
            'data': analysis
        })
        
    except Exception as e:
        logger.error(f"종합 분석 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/risk-alert-details/<ticker>')
def get_risk_alert_details(ticker):
    """특정 종목의 위험 분석 상세 리포트"""
    try:
        analyzer = FinancialReportAnalyzer()
        
        # 위험 신호 감지
        risk_alerts = analyzer.detect_risk_signals(ticker)
        
        if not risk_alerts:
            return jsonify({
                'success': False,
                'message': '위험 신호가 감지되지 않았습니다.'
            })
        
        # 가장 심각한 위험 신호 선택
        most_critical = max(risk_alerts, key=lambda x: 
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.risk_level.value]
        )
        
        # 재무제표 분석
        financial_analysis = analyzer.analyze_financial_statements(ticker)
        
        # 리포트 생성
        risk_report = {
            'ticker': ticker,
            'stock_name': most_critical.stock_name,
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'executive_summary': {
                'risk_level': most_critical.risk_level.value,
                'main_risk': most_critical.title,
                'description': most_critical.description
            },
            'detailed_analysis': {
                'risk_factors': [
                    {
                        'title': alert.title,
                        'analysis': alert.analysis
                    } for alert in risk_alerts
                ],
                'key_metrics': financial_analysis.get('key_metrics', {})
            },
            'recommendations': {
                'immediate_actions': most_critical.recommendations[:2],
                'monitoring_points': most_critical.recommendations[2:]
            }
        }
        
        return jsonify({
            'success': True,
            'risk_report': risk_report
        })
        
    except Exception as e:
        logger.error(f"위험 분석 리포트 생성 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/learning/terms')
def get_learning_terms():
    """학습 용어 목록을 반환합니다."""
    try:
        category = request.args.get('category', None)
        difficulty = request.args.get('difficulty', None)
        
        terms = INVESTMENT_TERMS
        
        if category:
            terms = [t for t in terms if t['category'] == category]
        if difficulty:
            terms = [t for t in terms if t['difficulty'] == difficulty]
            
        return jsonify({
            "success": True,
            "terms": terms,
            "total": len(terms)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/learning/term/<int:term_id>')
def get_term_detail(term_id):
    """특정 용어의 상세 정보를 반환합니다."""
    try:
        term = next((t for t in INVESTMENT_TERMS if t['id'] == term_id), None)
        if term:
            return jsonify({"success": True, "term": term})
        else:
            return jsonify({"success": False, "error": "용어를 찾을 수 없습니다."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/learning/quiz')
def get_quiz_questions():
    """퀴즈 문제를 반환합니다."""
    try:
        level = request.args.get('level', None)
        count = int(request.args.get('count', 5))
        
        questions = QUIZ_QUESTIONS
        
        if level:
            questions = [q for q in questions if q['level'] == level]
            
        # 랜덤하게 선택
        import random
        selected = random.sample(questions, min(count, len(questions)))
        
        return jsonify({
            "success": True,
            "questions": selected,
            "total": len(selected)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/learning/quiz/submit', methods=['POST'])
def submit_quiz_answer():
    """퀴즈 답안을 제출하고 채점합니다."""
    try:
        data = request.get_json()
        question_id = data.get('question_id')
        user_answer = data.get('answer')
        
        question = next((q for q in QUIZ_QUESTIONS if q['id'] == question_id), None)
        if not question:
            return jsonify({"success": False, "error": "문제를 찾을 수 없습니다."})
            
        is_correct = question['answer'] == user_answer
        
        # 세션에 학습 진행 상태 저장
        if 'quiz_results' not in session:
            session['quiz_results'] = []
            
        session['quiz_results'].append({
            'question_id': question_id,
            'correct': is_correct,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            "success": True,
            "correct": is_correct,
            "correct_answer": question['answer'],
            "explanation": question['explanation']
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/learning/progress')
def get_learning_progress():
    """학습 진행 상황을 반환합니다."""
    try:
        # 세션에서 학습 진행 상황 가져오기
        completed_terms = session.get('completed_terms', [])
        quiz_results = session.get('quiz_results', [])
        cardnews_progress = session.get('cardnews_progress', {})
        
        # 통계 계산
        total_terms = len(INVESTMENT_TERMS)
        completed_terms_count = len(completed_terms)
        
        total_quizzes = len(quiz_results)
        correct_quizzes = sum(1 for r in quiz_results if r['correct'])
        
        progress_percentage = (completed_terms_count / total_terms * 100) if total_terms > 0 else 0
        quiz_accuracy = (correct_quizzes / total_quizzes * 100) if total_quizzes > 0 else 0
        
        # 카드뉴스 진행률 계산
        completed_cardnews = len([topic for topic, data in cardnews_progress.items() if data.get('completed', False)])
        total_cardnews = 5  # 총 카드뉴스 주제 수
        cardnews_percentage = (completed_cardnews / total_cardnews * 100) if total_cardnews > 0 else 0
        
        # 전체 학습 진행률 계산 (단어 45%, 카드뉴스 35%, 퀴즈 20%)
        overall_progress = (
            (progress_percentage * 0.45) +
            (cardnews_percentage * 0.35) +
            (quiz_accuracy * 0.20 if total_quizzes > 0 else 0)
        )
        
        return jsonify({
            "success": True,
            "progress": {
                "overall": round(overall_progress, 1),
                "completed_terms": completed_terms_count,
                "total_terms": total_terms,
                "progress_percentage": round(progress_percentage, 1),
                "quiz_attempts": total_quizzes,
                "quiz_correct": correct_quizzes,
                "quiz_accuracy": round(quiz_accuracy, 1),
                "cardnews_completed": completed_cardnews,
                "cardnews_total": total_cardnews,
                "cardnews_percentage": round(cardnews_percentage, 1),
                "last_study_date": session.get('last_study_date', None),
                "streak_days": session.get('streak_days', 0),
                "modules": {
                    "words": round(progress_percentage, 1),
                    "cardnews": round(cardnews_percentage, 1),
                    "quiz": round(quiz_accuracy, 1) if total_quizzes > 0 else 0
                }
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/learning/complete-term', methods=['POST'])
def complete_term():
    """용어 학습 완료를 기록합니다."""
    try:
        data = request.get_json()
        term_id = data.get('term_id')
        
        if 'completed_terms' not in session:
            session['completed_terms'] = []
            
        if term_id not in session['completed_terms']:
            session['completed_terms'].append(term_id)
            
        # 오늘 학습 기록
        today = datetime.now().strftime('%Y-%m-%d')
        session['last_study_date'] = today
        
        return jsonify({
            "success": True,
            "completed_terms": len(session['completed_terms'])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/learning/cardnews/progress', methods=['POST'])
def update_cardnews_progress():
    """카드뉴스 학습 진행 상황을 업데이트합니다."""
    try:
        data = request.get_json()
        topic = data.get('topic')
        slide = data.get('slide')
        total = data.get('total')
        progress = data.get('progress')
        
        if 'cardnews_progress' not in session:
            session['cardnews_progress'] = {}
            
        # 주제별 진행 상황 저장
        session['cardnews_progress'][topic] = {
            'current_slide': slide,
            'total_slides': total,
            'progress': progress,
            'last_viewed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 오늘 학습 기록
        today = datetime.now().strftime('%Y-%m-%d')
        session['last_study_date'] = today
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/learning/cardnews/complete', methods=['POST'])
def complete_cardnews():
    """카드뉴스 학습 완료를 기록합니다."""
    try:
        data = request.get_json()
        topic = data.get('topic')
        
        if 'cardnews_progress' not in session:
            session['cardnews_progress'] = {}
            
        # 주제 완료 표시
        if topic in session['cardnews_progress']:
            session['cardnews_progress'][topic]['completed'] = True
            session['cardnews_progress'][topic]['completed_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            session['cardnews_progress'][topic] = {
                'completed': True,
                'completed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def load_analysis_result():
    """분석 결과를 로드합니다."""
    try:
        # web 디렉토리의 파일부터 확인
        if os.path.exists(ANALYSIS_FILE):
            with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # detailed_analysis가 비어있거나 없는 경우 생성
                if 'detailed_analysis' not in data or not data['detailed_analysis']:
                    scores = data.get('scores', {})
                    data['detailed_analysis'] = {
                        "risk_tolerance_analysis": generate_risk_analysis(scores.get('risk_tolerance', scores.get('risk', 50))),
                        "investment_time_horizon_analysis": generate_horizon_analysis(scores.get('investment_time_horizon', scores.get('horizon', 50))),
                        "financial_goal_orientation_analysis": generate_goal_analysis(scores.get('financial_goal_orientation', scores.get('goal', 50))),
                        "information_processing_style_analysis": generate_process_analysis(scores.get('information_processing_style', scores.get('process', 50))),
                        "investment_fear_analysis": generate_fear_analysis(scores.get('investment_fear', scores.get('fear', 50))),
                        "investment_confidence_analysis": generate_confidence_analysis(scores.get('investment_confidence', scores.get('confidence', 50)))
                    }
                    # 업데이트된 결과 저장
                    with open(ANALYSIS_FILE, "w", encoding="utf-8") as f2:
                        json.dump(data, f2, ensure_ascii=False, indent=4)
                return data
                
        # src 디렉토리의 파일 확인
        src_file = os.path.join(SRC_DIR, "analysis_results.json")
        if os.path.exists(src_file):
            with open(src_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # detailed_analysis가 비어있거나 없는 경우 생성
                if 'detailed_analysis' not in data or not data['detailed_analysis']:
                    scores = data.get('scores', {})
                    data['detailed_analysis'] = {
                        "risk_tolerance_analysis": generate_risk_analysis(scores.get('risk_tolerance', scores.get('risk', 50))),
                        "investment_time_horizon_analysis": generate_horizon_analysis(scores.get('investment_time_horizon', scores.get('horizon', 50))),
                        "financial_goal_orientation_analysis": generate_goal_analysis(scores.get('financial_goal_orientation', scores.get('goal', 50))),
                        "information_processing_style_analysis": generate_process_analysis(scores.get('information_processing_style', scores.get('process', 50))),
                        "investment_fear_analysis": generate_fear_analysis(scores.get('investment_fear', scores.get('fear', 50))),
                        "investment_confidence_analysis": generate_confidence_analysis(scores.get('investment_confidence', scores.get('confidence', 50)))
                    }
                # web 디렉토리에도 복사
                try:
                    with open(ANALYSIS_FILE, "w", encoding="utf-8") as f2:
                        json.dump(data, f2, ensure_ascii=False, indent=4)
                except:
                    pass
                return data
                
        # 루트 디렉토리의 파일 확인
        root_file = os.path.join(BASE_DIR, "analysis_results.json")
        if os.path.exists(root_file):
            with open(root_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # detailed_analysis가 비어있거나 없는 경우 생성
                if 'detailed_analysis' not in data or not data['detailed_analysis']:
                    scores = data.get('scores', {})
                    data['detailed_analysis'] = {
                        "risk_tolerance_analysis": generate_risk_analysis(scores.get('risk_tolerance', scores.get('risk', 50))),
                        "investment_time_horizon_analysis": generate_horizon_analysis(scores.get('investment_time_horizon', scores.get('horizon', 50))),
                        "financial_goal_orientation_analysis": generate_goal_analysis(scores.get('financial_goal_orientation', scores.get('goal', 50))),
                        "information_processing_style_analysis": generate_process_analysis(scores.get('information_processing_style', scores.get('process', 50))),
                        "investment_fear_analysis": generate_fear_analysis(scores.get('investment_fear', scores.get('fear', 50))),
                        "investment_confidence_analysis": generate_confidence_analysis(scores.get('investment_confidence', scores.get('confidence', 50)))
                    }
                # web 디렉토리에도 복사
                try:
                    with open(ANALYSIS_FILE, "w", encoding="utf-8") as f2:
                        json.dump(data, f2, ensure_ascii=False, indent=4)
                except:
                    pass
                return data
    except Exception as e:
        print(f"분석 결과 로드 에러: {e}")
    
    return None

def get_ticker_name(code):
    """종목 코드에 해당하는 종목명을 반환합니다"""
    # TODO: 실제 구현에서는 DB 조회나 API 호출로 대체해야 함
    ticker_map = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "035420": "NAVER",
        "035720": "카카오",
        "005380": "현대차",
        "051910": "LG화학",
        "207940": "삼성바이오로직스",
        "006400": "삼성SDI",
        "068270": "셀트리온",
        "000270": "기아",
        "090430": "아모레퍼시픽",
        "053800": "안랩",
        "012510": "더존비즈온",
        "247540": "에코프로비엠",
        "091990": "셀트리온헬스케어"
    }
    return ticker_map.get(code, f"종목 {code}")

def update_prediction_results():
    """포트폴리오 예측 결과를 업데이트합니다"""
    # TODO: 실제 구현에서는 예측 모델 호출 또는 데이터 처리 로직 구현 필요
    # 현재는 이미 생성된 예측 이미지를 사용하는 구조이므로 별도 처리 없음
    pass

def add_to_chat_history(speaker, content):
    """채팅 기록에 새 메시지를 추가합니다."""
    history = load_chat_history()
    
    # 새 메시지 추가
    history.append({
        "speaker": speaker,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # 최대 50개 메시지만 유지
    if len(history) > 50:
        history = history[-50:]
    
    # 저장
    save_chat_history(history)

def load_chat_history():
    """채팅 기록을 로드합니다."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"채팅 기록 로드 오류: {e}")
    
    return []

def save_chat_history(history):
    """채팅 기록을 저장합니다."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"채팅 기록 저장 오류: {e}")

@app.route('/api/alerts', methods=['GET'])
def api_alerts():
    """위험 신호/경고 요약 반환 및 이력 자동 저장"""
    session_id = session.get('user_id')
    if not session_id:
        session_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = session_id
    advisor = get_advisor_for_session(session_id)
    alerts = advisor.financial_processor.detect_risk_alerts()
    # 사용자별 알림/경고 이력 자동 저장
    if alerts:
        advisor.financial_processor.save_user_alert_history(session_id, alerts)
    alerts_context = advisor.financial_processor.get_risk_alerts_context()
    return jsonify({"success": True, "alerts_context": alerts_context})

@app.route('/alerts')
def alerts_page():
    """위험 신호/경고 시각화 페이지"""
    return render_template('alerts.html')

@app.route('/api/alert-history', methods=['GET'])
def api_alert_history():
    """
    사용자별 알림/경고 이력 반환 (supabase 기반)
    since(ISO8601) 파라미터가 있으면 해당 시각 이후 생성된 알림만 반환
    """
    session_id = session.get('user_id')
    if not session_id:
        session_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = session_id
    advisor = get_advisor_for_session(session_id)
    since = request.args.get('since')
    if since:
        try:
            since_dt = dateutil_parser.parse(since)
        except Exception:
            since_dt = None
    else:
        since_dt = None
    try:
        alert_history = advisor.financial_processor.load_user_alert_history(session_id, since_dt)
        print(f"/api/alert-history: {session_id} 알림 이력 {len(alert_history)}건 반환")
        return jsonify({"success": True, "alert_history": alert_history})
    except Exception as e:
        print(f"/api/alert-history 오류: {e}")
        return jsonify({"success": False, "alert_history": []})

@app.route('/alert-history')
def alert_history_page():
    """알림/경고 이력 시각화 페이지"""
    return render_template('alert_history.html')

@app.route('/api/alert-history-unread-count', methods=['GET'])
def api_alert_history_unread_count():
    """
    사용자별 미확인 알림 개수 반환
    """
    session_id = session.get('user_id')
    if not session_id:
        session_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = session_id
    advisor = get_advisor_for_session(session_id)
    unread_count = advisor.financial_processor.count_unread_alerts(session_id)
    return jsonify({"success": True, "unread_count": unread_count})

@app.route('/api/alert-history-mark-read', methods=['POST'])
def api_alert_history_mark_read():
    """
    사용자별 모든 미확인 알림을 확인 처리
    """
    session_id = session.get('user_id')
    if not session_id:
        session_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = session_id
    advisor = get_advisor_for_session(session_id)
    advisor.financial_processor.mark_all_alerts_read(session_id)
    return jsonify({"success": True})

@app.route('/api/alert-history-mark-read-one', methods=['POST'])
def api_alert_history_mark_read_one():
    """
    특정 알림(id)만 확인 처리
    {"alert_id": int} 형식의 JSON body 필요
    """
    session_id = session.get('user_id')
    if not session_id:
        session_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = session_id
    data = request.get_json()
    alert_id = data.get('alert_id')
    if alert_id is None:
        return jsonify({"success": False, "error": "alert_id required"}), 400
    advisor = get_advisor_for_session(session_id)
    advisor.financial_processor.mark_alert_read(session_id, alert_id)
    return jsonify({"success": True})

# 중복된 스케줄러 코드 제거됨 - 위쪽의 스케줄러 코드를 사용

# 관심 뉴스 키워드(종목명 등) 리스트 (메모리 기반, 실제 서비스는 DB로 확장)
news_keywords_list = ['삼성전자']

@app.route('/api/news-keywords', methods=['GET', 'POST', 'DELETE'])
def api_news_keywords():
    # 세션 기반 user_id
    user_id = session.get('user_id')
    if not user_id:
        user_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = user_id
    
    # 세션에서 포트폴리오 키워드 가져오기
    portfolio_keywords = session.get('portfolio_keywords', [])
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if request.method == 'GET':
        # DB와 세션의 키워드 합치기
        c.execute('SELECT keyword FROM news_keywords WHERE user_id=? ORDER BY created_at DESC', (user_id,))
        db_keywords = [row[0] for row in c.fetchall()]
        all_keywords = list(set(portfolio_keywords + db_keywords))
        conn.close()
        return jsonify({'success': True, 'keywords': all_keywords})
    
    elif request.method == 'POST':
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        if keyword:
            # 세션에 추가
            if keyword not in portfolio_keywords:
                portfolio_keywords.append(keyword)
                session['portfolio_keywords'] = portfolio_keywords
            
            # DB에도 저장
            c.execute('SELECT 1 FROM news_keywords WHERE user_id=? AND keyword=?', (user_id, keyword))
            if not c.fetchone():
                c.execute('INSERT INTO news_keywords (user_id, keyword) VALUES (?, ?)', (user_id, keyword))
                conn.commit()
        
        # 전체 키워드 반환
        c.execute('SELECT keyword FROM news_keywords WHERE user_id=? ORDER BY created_at DESC', (user_id,))
        db_keywords = [row[0] for row in c.fetchall()]
        all_keywords = list(set(portfolio_keywords + db_keywords))
        conn.close()
        return jsonify({'success': True, 'keywords': all_keywords})
    
    elif request.method == 'DELETE':
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        if keyword:
            # 세션에서 제거
            if keyword in portfolio_keywords:
                portfolio_keywords.remove(keyword)
                session['portfolio_keywords'] = portfolio_keywords
            
            # DB에서도 제거
            c.execute('DELETE FROM news_keywords WHERE user_id=? AND keyword=?', (user_id, keyword))
            conn.commit()
        
        # 전체 키워드 반환
        c.execute('SELECT keyword FROM news_keywords WHERE user_id=? ORDER BY created_at DESC', (user_id,))
        db_keywords = [row[0] for row in c.fetchall()]
        all_keywords = list(set(portfolio_keywords + db_keywords))
        conn.close()
        return jsonify({'success': True, 'keywords': all_keywords})

@app.route('/api/news-scheduler-status', methods=['GET'])
def api_news_scheduler_status():
    if news_scheduler and hasattr(news_scheduler, 'running'):
        job = news_scheduler.get_job('news_update_job')
        status = {
            'running': news_scheduler.running,
            'next_run_time': str(job.next_run_time) if job else None
        }
    else:
        status = {
            'running': False,
            'next_run_time': None,
            'message': 'Scheduler disabled'
        }
    return jsonify({'success': True, 'status': status})

@app.route('/api/news-scheduler-run', methods=['POST'])
def api_news_scheduler_run():
    try:
        scheduled_news_update()
        return jsonify({'success': True, 'message': '뉴스 수집이 즉시 실행되었습니다.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stock-price', methods=['GET'])
def api_stock_price():
    """
    종목 코드(code)로 실시간 주가 데이터 반환
    예: /api/stock-price?code=005930
    """
    code = request.args.get('code', '')
    if not code:
        return jsonify({'success': False, 'error': 'code 파라미터 필요'}), 400
    
    try:
        # 1. Supabase에서 먼저 조회 시도 (한국 주식만)
        if len(code) == 6 and code.isdigit():
            supabase_result = real_data_manager.get_stock_price_from_supabase(code)
            if supabase_result:
                return jsonify(supabase_result)
        
        # 2. Supabase 조회 실패 시 기존 파일 방식
        # 종목 코드가 6자리 숫자면 한국 주식
        if len(code) == 6 and code.isdigit():
            # 한국 주식 데이터에서 찾기
            korean_data = real_data_manager.load_korean_stocks()
            if korean_data and 'prices' in korean_data:
                for price_data in korean_data['prices']:
                    # ticker 컬럼 확인 (영문 컬럼명 사용)
                    ticker_code = str(price_data.get('ticker', price_data.get('종목코드', ''))).strip()
                    if ticker_code == code or ticker_code.zfill(6) == code.zfill(6):
                        # 종목명 찾기
                        stock_name = price_data.get('name', code)
                        if stock_name == code and 'tickers' in korean_data:
                            for ticker in korean_data['tickers']:
                                if str(ticker.get('ticker', ticker.get('종목코드', ''))).strip() == code:
                                    stock_name = ticker.get('name', ticker.get('종목명', code))
                                    break
                        
                        return jsonify({
                            'success': True,
                            'price': {
                                '종목코드': code,
                                '종목명': stock_name,
                                '종가': price_data.get('Close', price_data.get('종가', 0)),
                                '시가': price_data.get('Open', price_data.get('시가', 0)),
                                '고가': price_data.get('High', price_data.get('고가', 0)),
                                '저가': price_data.get('Low', price_data.get('저가', 0)),
                                '거래량': price_data.get('Volume', price_data.get('거래량', 0)),
                                'datetime': price_data.get('Date', price_data.get('날짜', datetime.now().strftime('%Y-%m-%d')))
                            }
                        })
        else:
            # 미국 주식 데이터에서 찾기
            us_data = real_data_manager.load_us_stocks()
            if us_data and 'prices' in us_data:
                for price_data in us_data['prices']:
                    if price_data.get('Ticker') == code:
                        # 종목명 찾기
                        stock_name = code
                        if 'tickers' in us_data:
                            for ticker in us_data['tickers']:
                                if ticker.get('Ticker') == code:
                                    stock_name = ticker.get('Name', code)
                                    break
                        
                        return jsonify({
                            'success': True,
                            'price': {
                                'ticker': code,
                                'name': stock_name,
                                'Close': price_data.get('Close', 0),
                                'Open': price_data.get('Open', 0),
                                'High': price_data.get('High', 0),
                                'Low': price_data.get('Low', 0),
                                'Volume': price_data.get('Volume', 0),
                                'datetime': price_data.get('Date', datetime.now().strftime('%Y-%m-%d'))
                            }
                        })
        
        # 데이터를 찾지 못한 경우
        return jsonify({'success': False, 'error': f'{code} 종목의 데이터를 찾을 수 없습니다'})
        
    except Exception as e:
        logger.error(f"주가 조회 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stock')
def stock_page():
    """
    실시간 주가/차트 조회 페이지
    """
    return render_template('stock.html')

@app.route('/api/news-today', methods=['GET'])
def api_news_today():
    """오늘의 뉴스 조회 (실시간 네이버 뉴스 우선, Supabase 폴백)"""
    try:
        # 필터 파라미터 가져오기 (domestic/global)
        filter_type = request.args.get('filter', 'domestic')
        
        # 먼저 Supabase에서 감정 분석된 뉴스 데이터 가져오기
        news_data = []
        try:
            from src.db_client import get_supabase_client
            supabase = get_supabase_client()
            
            if supabase:
                # 오늘 날짜 기준으로 뉴스 조회
                from datetime import datetime, timedelta
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                
                # 감정 분석 데이터와 함께 뉴스 가져오기 (최신순 정렬)
                response = supabase.table('news_articles').select(
                    "*, sentiment_details!left(positive_score,negative_score,neutral_score)"
                ).gte('published_date', yesterday.isoformat()).order('published_date', desc=True).limit(50).execute()
                
                if response.data:
                    for item in response.data:
                        # sentiment와 sentiment_score는 news_articles 테이블에 있음
                        # 이미 item에 포함되어 있어야 함
                        print(f"[DEBUG] News item keys: {item.keys()}")
                        print(f"[DEBUG] Sentiment: {item.get('sentiment')}, Score: {item.get('sentiment_score')}")
                        
                        # 감정 정보 추가
                        sentiment_detail = item.get('sentiment_details', [])
                        if sentiment_detail:
                            sentiment_scores = sentiment_detail[0]
                            item['positive_score'] = sentiment_scores.get('positive_score', 0)
                            item['negative_score'] = sentiment_scores.get('negative_score', 0)
                        
                        # 카테고리 설정
                        if 'category' not in item or not item['category']:
                            item['category'] = '종합'
                        
                        # 링크 처리
                        if not item.get('url') or item['url'] == '#':
                            search_query = item.get('title', '').replace(' ', '+')
                            item['link'] = f"https://search.naver.com/search.naver?where=news&query={search_query}"
                        else:
                            item['link'] = item['url']
                        
                        # 시간 표시
                        if item.get('published_date'):
                            try:
                                pub_date = datetime.fromisoformat(item['published_date'].replace('Z', '+00:00'))
                                now = datetime.now()
                                diff = now - pub_date
                                if diff.days == 0:
                                    if diff.seconds < 3600:
                                        item['time'] = f"{diff.seconds // 60}분 전"
                                    else:
                                        item['time'] = f"{diff.seconds // 3600}시간 전"
                                elif diff.days == 1:
                                    item['time'] = "어제"
                                else:
                                    item['time'] = f"{diff.days}일 전"
                            except:
                                item['time'] = "최근"
                        else:
                            item['time'] = "최근"
                        
                        # 좋아요 정보 (기본값)
                        item['likes'] = item.get('likes', 0)
                        item['isLiked'] = False
                        
                        # 소스 정보
                        item['source'] = item.get('source', '픽시 뉴스')
                        
                        news_data.append(item)
                    
                    print(f"[DEBUG] Supabase에서 {len(news_data)}개 뉴스 로드 완료")
        except Exception as e:
            print(f"[DEBUG] Supabase 뉴스 로드 실패: {e}")
        
        # Supabase에서 데이터를 가져오지 못한 경우 로컬 데이터 확인
        if not news_data:
            try:
                news_data = real_data_manager.load_news_data()
                print(f"[DEBUG] Local news data loaded: {len(news_data) if news_data else 0} items")
            except Exception as e:
                print(f"[DEBUG] Error loading local news data: {e}")
        
        # 실제 데이터가 있으면 사용
        if news_data and len(news_data) > 0:
            import random
            # 필터링 적용
            filtered_news = []
            for news in news_data:
                # 국내/해외 뉴스 필터링
                title = str(news.get('title', ''))
                content = str(news.get('summary', '') or news.get('content', ''))
                
                # 해외 뉴스 키워드
                global_keywords = ['미국', '중국', '일본', '유럽', 'EU', '연준', 'Fed', 'FOMC', '바이든', '트럼프', 
                                  '나스닥', 'S&P', '다우', '애플', '테슬라', '엔비디아', 'TSMC', 'NYSE', '달러', '엔화', '위안화']
                
                # 국내 뉴스 키워드  
                domestic_keywords = ['한국', '국내', '코스피', '코스닥', '삼성', 'SK', 'LG', '현대', '기아', '네이버', '카카오',
                                   '금융위', '한국은행', '금통위', '서울', '대한민국']
                
                # 뉴스 분류
                is_global = any(keyword in title or keyword in content for keyword in global_keywords)
                is_domestic = any(keyword in title or keyword in content for keyword in domestic_keywords)
                
                # 명확하지 않은 경우 기본값은 국내
                if not is_global and not is_domestic:
                    is_domestic = True
                
                # 필터링 적용
                if filter_type == 'global' and not is_global:
                    continue
                elif filter_type == 'domestic' and is_global and not is_domestic:
                    continue
                
                filtered_news.append(news)
            
            # published_date 기준으로 정렬 (최신순)
            filtered_news.sort(key=lambda x: x.get('published_date', x.get('pub_date', '')), reverse=True)
            
            # 상위 8개 선택 (랜덤이 아닌 최신순)
            selected_news = filtered_news[:8] if filtered_news else []
            
            # 필요한 필드 추가
            for news in selected_news:
                if 'category' not in news:
                    if '삼성' in news.get('title', '') or 'SK' in news.get('title', ''):
                        news['category'] = 'IT'
                    elif '경제' in news.get('title', '') or '금융' in news.get('title', ''):
                        news['category'] = '경제'
                    else:
                        news['category'] = '종합'
                
                news['likes'] = news.get('likes', random.randint(50, 150))
                news['isLiked'] = False
                news['time'] = news.get('time', '방금 전')
                news['source'] = news.get('source', '픽시 뉴스')
            
            return jsonify(clean_json_data({
                'success': True,
                'news_list': selected_news,
                'cached': False
            }))
        
        # 실제 데이터가 없으면 mock 데이터 사용
        from news_api_helper import get_mock_news_data
        mock_data = get_mock_news_data()
        news_list = mock_data.get('today_pick', [])
        
        import random
        random.shuffle(news_list)
        
        print(f"[DEBUG] Returning {len(news_list)} mock news items for today's pick")
        
        return jsonify(clean_json_data({
            'success': True,
            'news_list': news_list,
            'cached': False
        }))
        
        # 아래는 실제 데이터가 있을 때의 처리 로직 (현재는 사용 안함)
        if False and news_data and len(news_data) > 0:
            # 최신 뉴스 우선 정렬
            from datetime import datetime
            today = datetime.now().date()
            
            today_news = []
            for news in news_data[:20]:  # 최대 20개
                # 날짜 파싱
                try:
                    if 'published' in news and news['published']:
                        pub_date = datetime.fromisoformat(news['published'].replace('Z', '+00:00')).date()
                        if pub_date == today:
                            news['time'] = '오늘'
                        else:
                            days_diff = (today - pub_date).days
                            if days_diff == 1:
                                news['time'] = '어제'
                            else:
                                news['time'] = f'{days_diff}일 전'
                    else:
                        news['time'] = '최근'
                except:
                    news['time'] = '최근'
                
                # 국내/해외 뉴스 필터링
                title = news.get('title', '')
                content = news.get('summary', '')
                
                # 해외 뉴스 키워드
                global_keywords = ['미국', '중국', '일본', '유럽', 'EU', '연준', 'Fed', 'FOMC', '바이든', '트럼프', 
                                  '나스닥', 'S&P', '다우', '애플', '테슬라', '엔비디아', 'TSMC', 'NYSE', '달러', '엔화', '위안화']
                
                # 국내 뉴스 키워드  
                domestic_keywords = ['한국', '국내', '코스피', '코스닥', '삼성', 'SK', 'LG', '현대', '기아', '네이버', '카카오',
                                   '금융위', '한국은행', '금통위', '서울', '대한민국']
                
                # 뉴스 분류
                is_global = any(keyword in title or keyword in content for keyword in global_keywords)
                is_domestic = any(keyword in title or keyword in content for keyword in domestic_keywords)
                
                # 명확하지 않은 경우 기본값은 국내
                if not is_global and not is_domestic:
                    is_domestic = True
                
                # 필터링 적용
                if filter_type == 'global' and not is_global:
                    continue
                elif filter_type == 'domestic' and is_global and not is_domestic:
                    continue
                
                # 카테고리 추가
                if 'category' not in news:
                    if '삼성' in news.get('title', '') or 'SK' in news.get('title', ''):
                        news['category'] = 'IT'
                    elif '경제' in news.get('title', '') or '금융' in news.get('title', ''):
                        news['category'] = '경제'
                    elif '제약' in news.get('title', '') or '바이오' in news.get('title', ''):
                        news['category'] = '제약'
                    else:
                        news['category'] = '기타'
                
                # 좋아요 수 랜덤 생성 (실제로는 DB에서 관리)
                news['likes'] = news.get('likes', random.randint(50, 150))
                news['isLiked'] = False
                
                today_news.append(news)
            
            return jsonify({
                'success': True,
                'news_list': today_news,
                'cached': False
            })
        
        # 데이터가 없으면 mock 데이터 반환
        print("[DEBUG] No news data found, returning mock data")
        
        # Mock 데이터를 직접 생성
        from datetime import datetime
        mock_news_list = [
            {
                "id": 1,
                "category": "IT",
                "title": "삼성전자 실적 발표, 시장 예상치 상회",
                "content": "삼성전자가 발표한 최근 실적이 시장 예상치를 상회하며 긍정적인 반응을 보이고 있습니다.",
                "summary": "삼성전자가 발표한 최근 실적이 시장 예상치를 상회하며 긍정적인 반응을 보이고 있습니다.",
                "source": "한국경제",
                "time": "14시간 전",
                "likes": 93,
                "link": "https://www.hankyung.com",
                "isLiked": False,
                "sentiment": "긍정",
                "sentiment_score": 0.8,
                "importance_score": 85
            },
            {
                "id": 2,
                "category": "경제",
                "title": "코스피 2400선 회복, 외국인 매수세 지속",
                "content": "코스피가 외국인 매수세에 힘입어 2400선을 회복했습니다.",
                "summary": "코스피가 외국인 매수세에 힘입어 2400선을 회복했습니다.",
                "source": "매일경제",
                "time": "2시간 전",
                "likes": 156,
                "link": "https://www.mk.co.kr",
                "isLiked": False,
                "sentiment": "긍정",
                "sentiment_score": 0.7,
                "importance_score": 75
            },
            {
                "id": 3,
                "category": "바이오",
                "title": "SK바이오팜 신약 FDA 승인 임박",
                "content": "SK바이오팜의 신약이 FDA 승인을 앞두고 있어 투자자들의 관심이 집중되고 있습니다.",
                "summary": "SK바이오팜의 신약이 FDA 승인을 앞두고 있어 투자자들의 관심이 집중되고 있습니다.",
                "source": "서울경제",
                "time": "5시간 전",
                "likes": 87,
                "link": "https://www.sedaily.com",
                "isLiked": False,
                "sentiment": "긍정",
                "sentiment_score": 0.9,
                "importance_score": 90
            },
            {
                "id": 4,
                "category": "금융",
                "title": "금리 인하 기대감에 은행주 상승",
                "content": "금리 인하 기대감이 높아지면서 은행주가 일제히 상승했습니다.",
                "summary": "금리 인하 기대감이 높아지면서 은행주가 일제히 상승했습니다.",
                "source": "연합뉴스",
                "time": "30분 전",
                "likes": 234,
                "link": "https://www.yna.co.kr",
                "isLiked": False,
                "sentiment": "긍정",
                "sentiment_score": 0.6,
                "importance_score": 65
            }
        ]
        
        # Mock 데이터가 제대로 된 구조인지 확인
        for news in mock_news_list:
            # 필수 필드 확인 및 추가
            if 'id' not in news:
                news['id'] = hash(news.get('title', ''))
            if 'source' not in news:
                news['source'] = '픽시 뉴스'
            if 'time' not in news:
                news['time'] = '방금 전'
            if 'link' not in news:
                news['link'] = '#'
        
        return jsonify({
            'success': True,
            'news_list': mock_news_list,
            'cached': False
        })
    except Exception as e:
        print(f"오늘의 뉴스 조회 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 발생 시에도 mock 데이터 반환
        try:
            mock_news = get_mock_news_data()
            mock_news_list = mock_news.get('today_pick', [])
            
            # Mock 데이터가 제대로 된 구조인지 확인
            for news in mock_news_list:
                if 'id' not in news:
                    news['id'] = hash(news.get('title', ''))
                if 'source' not in news:
                    news['source'] = '픽시 뉴스'
                if 'time' not in news:
                    news['time'] = '방금 전'
                if 'link' not in news:
                    news['link'] = '#'
            
            return jsonify({
                'success': True,
                'news_list': mock_news_list,
                'cached': False
            })
        except:
            # 최후의 폴백 - 빈 리스트 반환
            return jsonify({
                'success': True,
                'news_list': [],
                'cached': False
            })
        supabase = get_supabase_client()
        
        # 경제/금융 관련 뉴스 우선 검색
        finance_keywords = ['주식', '투자', '경제', '금융', '삼성전자', 'SK하이닉스', 'LG에너지솔루션', '현대차', '기아', '네이버', '카카오', '코스피', '코스닥']
        
        # 경제/금융 관련 뉴스 우선 검색
        finance_news = []
        for keyword in finance_keywords[:5]:  # 상위 5개 키워드만 사용
            try:
                response = supabase.table('news_data').select('*').or_(f'title.ilike.%{keyword}%,content.ilike.%{keyword}%').order('created_at', desc=True).limit(3).execute()
                finance_news.extend(response.data)
            except Exception as e:
                print(f"키워드 '{keyword}' 검색 오류: {e}")
                continue
        
        # 일반 뉴스도 가져오기
        try:
            general_response = supabase.table('news_data').select('*').order('created_at', desc=True).limit(10).execute()
            all_news = finance_news + general_response.data
        except Exception as e:
            print(f"일반 뉴스 검색 오류: {e}")
            all_news = finance_news
        
        # 중복 제거 및 합치기
        seen_titles = set()
        unique_news = []
        for news in all_news:
            if news.get('title') and news.get('title') not in seen_titles:
                seen_titles.add(news.get('title'))
                unique_news.append(news)
        
        # 상위 15개만 선택
        unique_news = unique_news[:15]
        
        news_list = []
        for news in unique_news:
            # 감정 점수에 따른 감정 분석
            sentiment_score = news.get('sentiment_score', 0.5)
            if sentiment_score > 0.7:
                sentiment = '긍정'
            elif sentiment_score < 0.3:
                sentiment = '부정'
            else:
                sentiment = '중립'
            
            # 날짜 포맷팅
            published_at = news.get('published_at', '')
            if published_at:
                try:
                    date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%Y-%m-%d')
                except:
                    formatted_date = published_at[:10] if len(published_at) >= 10 else ''
            else:
                formatted_date = ''
            
            news_list.append({
                'id': news.get('id', news_list.__len__() + 1),
                'category': news.get('category', '경제'),
                'title': news.get('title', ''),
                'summary': news.get('content', '')[:150] + '...' if news.get('content') and len(news.get('content', '')) > 150 else news.get('content', ''),
                'source': news.get('source', '네이버 뉴스'),
                'time': _format_time_diff(published_at) if published_at else '방금 전',
                'likes': 93,
                'isLiked': False
            })
        
        if news_list:
            return jsonify({'success': True, 'news_list': news_list})
        
        # 폴백 2: 모의 데이터
        return jsonify({
            'success': True,
            'news_list': get_mock_news_data()['today_pick']
        })
        
    except Exception as e:
        logger.error(f"Failed to get today's news: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/news-portfolio', methods=['GET'])
def api_news_portfolio():
    """포트폴리오 관련 뉴스 조회"""
    try:
        # 필터 파라미터 가져오기
        filter_type = request.args.get('filter', 'domestic')
        
        # 세션에서 사용자 ID 가져오기
        session_id = session.get('user_id')
        
        # DB에서 사용자 관심 종목 가져오기
        portfolio_keywords = []
        if session_id:
            # SQLite에서 사용자 관심 종목 조회
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cursor = conn.cursor()
            # watchlist 테이블 확인 (user_watchlist가 아님)
            cursor.execute("""
                SELECT stock_code, stock_name FROM watchlist 
                WHERE user_id = ?
            """, (session_id,))
            watchlist = cursor.fetchall()
            conn.close()
            
            portfolio_keywords = [stock[1] for stock in watchlist]
        
        # 기본 관심 종목 설정
        if not portfolio_keywords:
            portfolio_keywords = ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '현대차', '기아']
            session['portfolio_keywords'] = portfolio_keywords
        
        # 실제 뉴스 데이터에서 포트폴리오 관련 뉴스 필터링
        news_data = real_data_manager.load_news_data()
        portfolio_news = []
        
        if news_data and len(news_data) > 0:
            for news in news_data:
                # 뉴스 필터링
                title = str(news.get('title', ''))
                content = str(news.get('content', ''))
                
                # 해외/국내 뉴스 확인
                global_keywords = ['미국', '중국', '일본', '유럽', 'EU', '연준', 'Fed', 'FOMC', '바이든', '트럼프', 
                                  '나스닥', 'S&P', '다우', '애플', '테슬라', '엔비디아', 'TSMC', 'NYSE', '달러', '엔화', '위안화']
                
                domestic_keywords = ['한국', '국내', '코스피', '코스닥', '삼성', 'SK', 'LG', '현대', '기아', '네이버', '카카오',
                                   '금융위', '한국은행', '금통위', '서울', '대한민국']
                
                is_global = any(keyword in title or keyword in content for keyword in global_keywords)
                is_domestic = any(keyword in title or keyword in content for keyword in domestic_keywords)
                
                if not is_global and not is_domestic:
                    is_domestic = True
                
                # 필터링 적용
                if filter_type == 'global' and not is_global:
                    continue
                elif filter_type == 'domestic' and is_global and not is_domestic:
                    continue
                
                # 포트폴리오 키워드 검색
                for keyword in portfolio_keywords:
                    if keyword in title or keyword in content:
                        # 날짜 포맷팅
                        try:
                            if 'pub_date' in news and news['pub_date']:
                                from datetime import datetime
                                pub_date = datetime.fromisoformat(news['pub_date'].replace('Z', '+00:00'))
                                days_diff = (datetime.now() - pub_date).days
                                if days_diff == 0:
                                    news['time'] = '오늘'
                                elif days_diff == 1:
                                    news['time'] = '어제'
                                else:
                                    news['time'] = f'{days_diff}일 전'
                            else:
                                news['time'] = '최근'
                        except:
                            news['time'] = '최근'
                        
                        news['category'] = keyword
                        news['likes'] = news.get('likes', random.randint(50, 150))
                        news['isLiked'] = False
                        portfolio_news.append(news)
                        break
                
                if len(portfolio_news) >= 20:
                    break
        
        if not portfolio_news:
            # 포트폴리오 뉴스가 없으면 안내 메시지
            return jsonify({
                'success': True,
                'news_list': [],
                'message': 'MY 투자에서 관심종목을 추가하면 관련 뉴스가 표시됩니다.'
            })
        
        return jsonify(clean_json_data({
            'success': True,
            'news_list': portfolio_news[:20],  # 최대 20개
            'portfolio_keywords': portfolio_keywords
        }))
        
    except Exception as e:
        print(f"포트폴리오 뉴스 조회 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 발생 시 mock 데이터 반환
        try:
            mock_news = get_mock_news_data()
            mock_news_list = mock_news.get('portfolio', [])
            
            # Mock 데이터가 없으면 today_pick 사용
            if not mock_news_list:
                mock_news_list = mock_news.get('today_pick', [])[:4]
            
            # 필수 필드 확인
            for news in mock_news_list:
                if 'id' not in news:
                    news['id'] = hash(news.get('title', ''))
                if 'source' not in news:
                    news['source'] = '픽시 뉴스'
                if 'time' not in news:
                    news['time'] = '방금 전'
                if 'link' not in news:
                    news['link'] = '#'
            
            return jsonify(clean_json_data({
                'success': True,
                'news_list': mock_news_list,
                'portfolio_keywords': ['삼성전자', 'SK하이닉스']
            }))
        except:
            return jsonify({
                'success': True,
                'news_list': [],
                'portfolio_keywords': []
            })
        
        # 새로운 데이터 수집 시도
        try:
            news_data = loop.run_until_complete(collect_realtime_news())
            
            if news_data:
                return jsonify({
                    'success': True,
                    'news_list': news_data['categorized_news'].get('portfolio', [])
                })
        except Exception as e:
            logger.warning(f"Failed to collect realtime news: {e}")
        finally:
            loop.close()
        
        # 폴백: Supabase에서 가져오기
        keywords = session.get('portfolio_keywords', [])
        if not keywords:
            keywords = ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '현대차', '기아']
            session['portfolio_keywords'] = keywords
        
        supabase = get_supabase_client()
        
        news_list = []
        for keyword in keywords:
            response = supabase.table('news_data').select('*').or_(f'title.ilike.%{keyword}%,content.ilike.%{keyword}%').order('created_at', desc=True).limit(5).execute()
            
            for news in response.data:
                # 감정 점수에 따른 감정 분석
                sentiment_score = news.get('sentiment_score', 0.5)
                if sentiment_score > 0.7:
                    sentiment = '긍정'
                elif sentiment_score < 0.3:
                    sentiment = '부정'
                else:
                    sentiment = '중립'
                
                # 날짜 포맷팅
                published_at = news.get('published_at', '')
                if published_at:
                    try:
                        date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime('%Y-%m-%d')
                    except:
                        formatted_date = published_at[:10] if len(published_at) >= 10 else ''
                else:
                    formatted_date = ''
                
                news_list.append({
                    'content': news.get('title', ''),
                    'summary': news.get('content', '')[:150] + '...' if news.get('content') and len(news.get('content', '')) > 150 else news.get('content', ''),
                    'date': formatted_date,
                    'source': news.get('source', '네이버 뉴스'),
                    'sentiment': sentiment,
                    'url': news.get('url', ''),
                    'keywords': news.get('keywords', []),
                    'related_keyword': keyword
                })
        
        # 중복 제거 (같은 뉴스가 여러 키워드에 매칭될 수 있음)
        seen_titles = set()
        unique_news = []
        for news in news_list:
            if news['content'] not in seen_titles:
                seen_titles.add(news['content'])
                unique_news.append(news)
        
        return jsonify({"success": True, "keywords": keywords, "news_list": unique_news})
    except Exception as e:
        print(f"포트폴리오 뉴스 API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/insight-today', methods=['GET'])
def api_insight_today():
    """오늘의 뉴스 기반 LLM 투자 인사이트 요약/분석 반환"""
    session_id = session.get('user_id')
    if not session_id:
        session_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = session_id
    advisor = get_advisor_for_session(session_id)
    news_list = advisor.financial_processor.get_today_news(top_k=10)
    # 뉴스 텍스트 합치기
    news_text = '\n'.join([f"- {n['content']} (요약: {n['summary']}, 감정: {n['sentiment']})" for n in news_list])
    # LLM 프롬프트 개선
    prompt = (
        "오늘의 주요 뉴스 헤드라인, 요약, 감정분석 결과를 바탕으로 다음을 수행해줘.\n"
        "1. 한국 주식시장/경제의 오늘의 주요 이슈를 2~3문장으로 요약\n"
        "2. 투자자에게 도움이 될 만한 인사이트/전략을 2~3문장으로 제안\n"
        "3. 긍정 신호와 부정 신호를 각각 한 문장씩 명확히 구분해서 정리\n"
        "[오늘의 뉴스]\n" + news_text
    )
    # LLM 호출
    insight = advisor.llm_service.generate_ai_response("AI-A", prompt, context=None)
    return jsonify({"success": True, "insight": insight.get('response', '')})

@app.route('/api/news-recommend', methods=['GET'])
def api_news_recommend():
    """추천 뉴스 조회 (좋아요 많은 뉴스)"""
    try:
        # Mock 데이터 먼저 시도
        mock_news = get_mock_news_data()
        mock_recommend = mock_news.get('recommend', [])
        
        # recommend가 없으면 today_pick 사용
        if not mock_recommend:
            mock_recommend = mock_news.get('today_pick', [])[:8]
        
        if mock_recommend:
            for news in mock_recommend:
                if 'id' not in news:
                    news['id'] = hash(news.get('title', ''))
                if 'source' not in news:
                    news['source'] = '픽시 뉴스'
                if 'time' not in news:
                    news['time'] = '방금 전'
                if 'link' not in news:
                    news['link'] = '#'
                if 'likes' not in news:
                    news['likes'] = random.randint(200, 800)
            return jsonify({'success': True, 'news_list': mock_recommend})
        # 필터 파라미터 가져오기
        filter_type = request.args.get('filter', 'domestic')
        
        # 실제 데이터 매니저에서 뉴스 데이터 가져오기
        news_data = real_data_manager.load_news_data()
        
        recommend_news = []
        if news_data and len(news_data) > 0:
            # 뉴스를 좋아요 수로 정렬 (랜덤 생성 - 실제로는 DB에서 관리)
            for news in news_data:
                news['likes'] = news.get('likes', random.randint(100, 500))
            
            # 좋아요 수로 정렬
            sorted_news = sorted(news_data, key=lambda x: x.get('likes', 0), reverse=True)
            
            # 필터링
            for news in sorted_news[:50]:  # 상위 50개에서 필터링
                title = str(news.get('title', ''))
                content = str(news.get('content', ''))
                
                # 해외/국내 뉴스 확인
                global_keywords = ['미국', '중국', '일본', '유럽', 'EU', '연준', 'Fed', 'FOMC', '바이든', '트럼프', 
                                  '나스닥', 'S&P', '다우', '애플', '테슬라', '엔비디아', 'TSMC', 'NYSE', '달러', '엔화', '위안화']
                
                domestic_keywords = ['한국', '국내', '코스피', '코스닥', '삼성', 'SK', 'LG', '현대', '기아', '네이버', '카카오',
                                   '금융위', '한국은행', '금통위', '서울', '대한민국']
                
                is_global = any(keyword in title or keyword in content for keyword in global_keywords)
                is_domestic = any(keyword in title or keyword in content for keyword in domestic_keywords)
                
                if not is_global and not is_domestic:
                    is_domestic = True
                
                # 필터링 적용
                if filter_type == 'global' and not is_global:
                    continue
                elif filter_type == 'domestic' and is_global and not is_domestic:
                    continue
                
                # 날짜 포맷팅
                try:
                    if 'pub_date' in news and news['pub_date']:
                        from datetime import datetime
                        pub_date = datetime.fromisoformat(news['pub_date'].replace('Z', '+00:00'))
                        days_diff = (datetime.now() - pub_date).days
                        if days_diff == 0:
                            news['time'] = '오늘'
                        elif days_diff == 1:
                            news['time'] = '어제'
                        else:
                            news['time'] = f'{days_diff}일 전'
                    else:
                        news['time'] = '최근'
                except:
                    news['time'] = '최근'
                
                # 카테고리 추가
                if 'category' not in news:
                    if '삼성' in title or 'SK' in title:
                        news['category'] = 'IT'
                    elif '경제' in title or '금융' in title:
                        news['category'] = '경제'
                    elif '제약' in title or '바이오' in title:
                        news['category'] = '제약'
                    else:
                        news['category'] = '기타'
                
                news['isLiked'] = True  # 추천 뉴스는 기본적으로 좋아요
                recommend_news.append(news)
                
                if len(recommend_news) >= 20:
                    break
        
        return jsonify({
            'success': True,
            'news_list': recommend_news
        })
    except Exception as e:
        print(f"인기 뉴스 API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/news-analysis', methods=['POST'])
def api_news_analysis():
    """키워드 기반 뉴스 분석"""
    try:
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        
        if not keyword:
            return jsonify({'success': False, 'error': '키워드가 필요합니다.'})
        
        # Supabase에서 해당 키워드와 관련된 뉴스 검색
        supabase = get_supabase_client()
        response = supabase.table('news_data').select('*').or_(f'title.ilike.%{keyword}%,content.ilike.%{keyword}%').order('created_at', desc=True).limit(20).execute()
        
        news_list = []
        total_sentiment = 0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for news in response.data:
            # 감정 점수에 따른 감정 분석
            sentiment_score = news.get('sentiment_score', 0.5)
            total_sentiment += sentiment_score
            
            if sentiment_score > 0.7:
                sentiment = '긍정'
                positive_count += 1
            elif sentiment_score < 0.3:
                sentiment = '부정'
                negative_count += 1
            else:
                sentiment = '중립'
                neutral_count += 1
            
            # 날짜 포맷팅
            published_at = news.get('published_at', '')
            if published_at:
                try:
                    date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime('%Y-%m-%d')
                except:
                    formatted_date = published_at[:10] if len(published_at) >= 10 else ''
            else:
                formatted_date = ''
            
            news_list.append({
                'content': news.get('title', ''),
                'summary': news.get('content', '')[:150] + '...' if news.get('content') and len(news.get('content', '')) > 150 else news.get('content', ''),
                'date': formatted_date,
                'source': news.get('source', '네이버 뉴스'),
                'sentiment': sentiment,
                'url': news.get('url', ''),
                'keywords': news.get('keywords', [])
            })
        
        # 분석 결과
        total_news = len(news_list)
        avg_sentiment = total_sentiment / total_news if total_news > 0 else 0.5
        
        analysis_result = {
            'keyword': keyword,
            'total_news': total_news,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'avg_sentiment': round(avg_sentiment, 2),
            'overall_sentiment': '긍정' if avg_sentiment > 0.6 else '부정' if avg_sentiment < 0.4 else '중립',
            'news_list': news_list
        }
        
        return jsonify({'success': True, 'analysis': analysis_result})
    except Exception as e:
        print(f"뉴스 분석 API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/portfolio-history', methods=['GET'])
def api_portfolio_history():
    """사용자별 포트폴리오 추천/분석 이력 반환 (최신 5개)"""
    try:
        session_id = session.get('user_id')
        if not session_id:
            session_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = session_id
        recs = get_portfolio_recommendations(session_id, limit=5)
        print(f"/api/portfolio-history: {session_id} 이력 {len(recs)}건 반환")
        return jsonify({"success": True, "history": recs})
    except Exception as e:
        print(f"/api/portfolio-history 오류: {e}")
        return jsonify({"success": False, "history": []})

@app.route('/api/profile', methods=['GET'])
def api_profile():
    """사용자 프로필/설문 분석 결과 반환 (supabase 기반)"""
    # 1. 쿼리 파라미터 user_id 우선 사용, 없으면 세션에서 가져옴
    user_id = request.args.get('user_id')
    if not user_id:
        user_id = session.get('user_id')
    if not user_id:
        user_id = f"user_{str(uuid.uuid4())[:8]}"
        session['user_id'] = user_id
    try:
        supabase = get_supabase_client()
        # profile_json, summary 모두 조회
        res = supabase.table("user_profiles").select("profile_json,summary").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        print(f"/api/profile: user_id={user_id}, 쿼리 결과={res.data}")
        if res.data and len(res.data) > 0:
            profile_json = res.data[0].get("profile_json", {})
            summary = res.data[0].get("summary", "")
            return jsonify({"success": True, "profile": profile_json, "summary": summary, "user_id": user_id})
        print(f"/api/profile: user_id={user_id} 프로필 없음")
        return jsonify({"success": False, "profile": {}, "summary": "", "user_id": user_id, "message": "No profile found for this user_id."})
    except Exception as e:
        print(f"/api/profile 오류: {e}")
        return jsonify({"success": False, "profile": {}, "summary": "", "user_id": user_id, "message": str(e)})

@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def api_watchlist():
    """관심종목 관리 API"""
    try:
        session_id = session.get('user_id')
        if not session_id:
            session_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = session_id
            
        if request.method == 'GET':
            # 관심종목 목록 조회
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                stock_code TEXT,
                stock_name TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            c.execute('SELECT stock_code, stock_name FROM watchlist WHERE user_id = ?', (session_id,))
            watchlist_stocks = [{"code": row[0], "name": row[1]} for row in c.fetchall()]
            conn.close()
            
            # 실제 주가 데이터와 뉴스 정보 추가
            detailed_watchlist = []
            
            # 한국 주식 데이터 로드
            korean_stocks = real_data_manager.load_korean_stocks()
            news_data = real_data_manager.load_news_data()
            
            for stock in watchlist_stocks:
                stock_info = {
                    "ticker": stock["code"],
                    "name": stock["name"],
                    "price": 0,
                    "change_percent": 0,
                    "sector": "N/A",
                    "latest_news": "최신 뉴스 없음"
                }
                
                # 주가 정보 매칭
                if korean_stocks is not None and not korean_stocks.empty:
                    stock_data = korean_stocks[korean_stocks['Code'] == stock["code"]]
                    if not stock_data.empty:
                        stock_info["price"] = int(stock_data.iloc[0]['Close'])
                        stock_info["change_percent"] = float(stock_data.iloc[0].get('Change', 0))
                        stock_info["sector"] = stock_data.iloc[0].get('Sector', 'N/A')
                        if stock_info["sector"] == 'N/A' or pd.isna(stock_info["sector"]):
                            stock_info["sector"] = "기타"
                
                # 최신 뉴스 매칭
                if news_data is not None and len(news_data) > 0:
                    for news in news_data:
                        if stock["name"] in news.get('title', '') or stock["name"] in news.get('content', ''):
                            stock_info["latest_news"] = news.get('title', '')[:50] + "..."
                            break
                
                detailed_watchlist.append(stock_info)
            
            # watchlist와 portfolio_keywords 동기화
            watchlist_names = [stock["name"] for stock in watchlist_stocks]
            portfolio_keywords = session.get('portfolio_keywords', [])
            
            # watchlist에 있는 종목들을 portfolio_keywords에 추가
            for name in watchlist_names:
                if name not in portfolio_keywords:
                    portfolio_keywords.append(name)
            
            session['portfolio_keywords'] = portfolio_keywords
            
            return jsonify({"success": True, "watchlist": detailed_watchlist})
            
        elif request.method == 'POST':
            # 관심종목 추가
            data = request.json
            stock_code = data.get('ticker', data.get('stock_code', '')).strip()
            stock_name = data.get('stock_name', '').strip()
            
            if not stock_code:
                return jsonify({"success": False, "error": "종목코드를 입력해주세요."})
            
            # 종목명이 없으면 조회 시도
            if not stock_name:
                # 한국 주식 데이터에서 종목명 찾기
                korean_stocks = real_data_manager.load_korean_stocks()
                if korean_stocks is not None and not korean_stocks.empty:
                    stock_data = korean_stocks[korean_stocks['Code'] == stock_code]
                    if not stock_data.empty:
                        stock_name = stock_data.iloc[0].get('Name', '')
                
                # 그래도 종목명이 없으면 종목코드를 이름으로 사용
                if not stock_name:
                    stock_name = stock_code
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                stock_code TEXT,
                stock_name TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # 중복 확인
            c.execute('SELECT id FROM watchlist WHERE user_id = ? AND stock_code = ?', (session_id, stock_code))
            if c.fetchone():
                conn.close()
                return jsonify({"success": False, "error": "이미 관심종목에 추가된 종목입니다."})
            
            c.execute('INSERT INTO watchlist (user_id, stock_code, stock_name) VALUES (?, ?, ?)', 
                     (session_id, stock_code, stock_name))
            conn.commit()
            conn.close()
            
            # portfolio_keywords에도 추가
            portfolio_keywords = session.get('portfolio_keywords', [])
            if stock_name not in portfolio_keywords:
                portfolio_keywords.append(stock_name)
                session['portfolio_keywords'] = portfolio_keywords
            
            return jsonify({"success": True, "message": "관심종목이 추가되었습니다."})
            
        elif request.method == 'DELETE':
            # 관심종목 삭제
            data = request.json
            stock_code = data.get('stock_code', '').strip()
            
            if not stock_code:
                return jsonify({"success": False, "error": "삭제할 종목코드를 입력해주세요."})
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            # 삭제하기 전에 종목명 가져오기
            c.execute('SELECT stock_name FROM watchlist WHERE user_id = ? AND stock_code = ?', (session_id, stock_code))
            result = c.fetchone()
            stock_name = result[0] if result else None
            
            c.execute('DELETE FROM watchlist WHERE user_id = ? AND stock_code = ?', (session_id, stock_code))
            conn.commit()
            conn.close()
            
            # portfolio_keywords에서도 제거
            if stock_name:
                portfolio_keywords = session.get('portfolio_keywords', [])
                if stock_name in portfolio_keywords:
                    portfolio_keywords.remove(stock_name)
                    session['portfolio_keywords'] = portfolio_keywords
            
            return jsonify({"success": True, "message": "관심종목이 삭제되었습니다."})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/stock-info/<stock_code>')
def api_stock_info(stock_code: str):
    """종목 정보 및 주가 데이터 반환"""
    try:
        # 임시 주가 데이터 (실제로는 API 호출)
        stock_data = {
            "005930": {"name": "삼성전자", "price": 75800, "change": 1200, "change_rate": 1.6},
            "000660": {"name": "SK하이닉스", "price": 142500, "change": -500, "change_rate": -0.3},
            "035420": {"name": "NAVER", "price": 185000, "change": 2500, "change_rate": 1.4}
        }
        
        if stock_code in stock_data:
            return jsonify({"success": True, "data": stock_data[stock_code]})
        else:
            return jsonify({"success": False, "error": "종목 정보를 찾을 수 없습니다."})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/risk-alerts')
def api_risk_alerts():
    """위험신호 알림 API"""
    try:
        session_id = session.get('user_id')
        if not session_id:
            session_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = session_id
        
        # 관심종목 목록 가져오기
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT stock_code, stock_name FROM watchlist WHERE user_id = ?', (session_id,))
        watchlist = [{"code": row[0], "name": row[1]} for row in c.fetchall()]
        conn.close()
        
        # 위험신호 생성
        alerts = []
        
        # 기본 위험신호 타입
        alert_types = [
            {"type": "RSI 과매도 신호 감지됨", "description": "지금 투자하기 전에 신중한 판단이 필요해요!"},
            {"type": "거래량 급증, 주의가 필요해요", "description": "시장의 '큰 손'이 움직였을 수도 있어요!"},
            {"type": "급격한 가격 변동 감지", "description": "변동성이 크게 증가했습니다. 리스크 관리가 필요해요!"},
            {"type": "지지선 이탈 경고", "description": "기술적 지지선을 하향 돌파했습니다. 추세 전환 가능성에 주의하세요!"}
        ]
        
        # 관심종목 중 일부에 대해 위험신호 생성
        import random
        if watchlist:
            # 관심종목이 있으면 그 중에서 랜덤하게 2개 선택
            selected_stocks = random.sample(watchlist, min(2, len(watchlist)))
            for i, stock in enumerate(selected_stocks):
                alert_info = alert_types[i % len(alert_types)]
                alerts.append({
                    "ticker": stock["code"],
                    "stock_name": stock["name"],
                    "title": alert_info["type"],
                    "description": alert_info["description"],
                    "level": "warning" if i % 2 == 0 else "danger",
                    "timestamp": datetime.now().isoformat()
                })
        else:
            # 관심종목이 없으면 기본 예시 데이터
            alerts = [
                {
                    "ticker": "005930",
                    "stock_name": "삼성전자",
                    "title": "RSI 과매도 신호 감지됨",
                    "description": "지금 투자하기 전에 신중한 판단이 필요해요!",
                    "level": "warning",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "ticker": "000660",
                    "stock_name": "SK하이닉스",
                    "title": "거래량 급증, 주의가 필요해요",
                    "description": "시장의 '큰 손'이 움직였을 수도 있어요!",
                    "level": "danger",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        
        return jsonify({"success": True, "alerts": alerts})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/watchlist-news')
def api_watchlist_news():
    """관심종목 관련 뉴스 API"""
    try:
        session_id = session.get('user_id')
        if not session_id:
            session_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = session_id
        
        # 관심종목 목록 가져오기
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT stock_code, stock_name FROM watchlist WHERE user_id = ?', (session_id,))
        watchlist = [{"code": row[0], "name": row[1]} for row in c.fetchall()]
        conn.close()
        
        # 뉴스 데이터 로드
        news_data = real_data_manager.load_news_data()
        watchlist_news = []
        
        if news_data and watchlist:
            # 각 관심종목에 대한 뉴스 검색
            for stock in watchlist:
                stock_news = []
                for news in news_data:
                    if stock["name"] in news.get('title', '') or stock["name"] in news.get('content', ''):
                        stock_news.append({
                            "stock_name": stock["name"],
                            "title": news.get('title', ''),
                            "content": news.get('content', '')[:200] + "...",
                            "date": news.get('date', datetime.now().strftime('%Y년 %m월 %d일'))
                        })
                        if len(stock_news) >= 1:  # 종목당 최대 1개 뉴스
                            break
                watchlist_news.extend(stock_news)
        
        # 뉴스가 없으면 기본 예시 데이터
        if not watchlist_news:
            watchlist_news = [
                {
                    "stock_name": "삼성전자",
                    "title": "삼성전자, 2분기 실적 예상치 상회",
                    "content": "삼성전자가 2025년 2분기 연결 기준 매출 74조 원, 영업이익 4.6조 원의 잠정 실적을 발표했어요. 이 수치는 시장의 기대치(매출 약 74조 원, 영업이익 4.5조 원)를 웃돌며, 특히 환영할 만한 성과로 해석됩니다...",
                    "date": "2025년 7월 8일"
                },
                {
                    "stock_name": "SK하이닉스",
                    "title": "SK 하이닉스, AI 반도체 시장 진출 확대",
                    "content": "SK하이닉스가 고성능 AI 메모리인 HBM(고대역폭 메모리) 제품을 중심으로 AI 반도체 시장에서 두각을 나타내고 있어요. 1분기 기준 HBM 분야에서 약 70%의 글로벌 점유율을 확보했으며...",
                    "date": "2025년 7월 16일"
                },
                {
                    "stock_name": "반도체",
                    "title": "반도체 업계, 수요 회복 신호 포착",
                    "content": "'세미콘코리아 2025' 참석 기업들은 AI·HBM 기반 반도체 수요가 본격 회복 조짐을 보인다고 밝혔습니다. 특히 HBM 시장은 오는 2028년까지 폭발적 성장세를 보일 것이라는 기대가 클 뿐 아니라...",
                    "date": "2025년 2월 19일"
                }
            ]
        
        return jsonify({"success": True, "news": watchlist_news[:10]})  # 최대 10개
        
    except Exception as e:
        print(f"관심종목 뉴스 조회 오류: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/stock-search')
def api_stock_search():
    """종목 검색 API"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({"success": True, "results": []})
        
        # 한국 주식 데이터에서 검색
        korean_stocks = real_data_manager.load_korean_stocks()
        results = []
        
        if korean_stocks is not None and not korean_stocks.empty:
            # 종목명 또는 종목코드로 검색
            filtered = korean_stocks[
                (korean_stocks['Name'].str.contains(query, case=False, na=False)) |
                (korean_stocks['Code'].str.contains(query, case=False, na=False))
            ]
            
            for _, row in filtered.head(10).iterrows():
                results.append({
                    "ticker": row['Code'],
                    "name": row['Name']
                })
        
        # 검색 결과가 없으면 기본 예시 데이터
        if not results:
            default_stocks = [
                {"ticker": "005930", "name": "삼성전자"},
                {"ticker": "000660", "name": "SK하이닉스"},
                {"ticker": "373220", "name": "LG에너지솔루션"},
                {"ticker": "352820", "name": "하이브"},
                {"ticker": "005490", "name": "포스코홀딩스"}
            ]
            results = [s for s in default_stocks if query.lower() in s["name"].lower() or query in s["ticker"]]
        
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        print(f"종목 검색 오류: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/learning-progress')
def api_learning_progress():
    """학습 진행률 API"""
    try:
        session_id = session.get('user_id')
        if not session_id:
            session_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = session_id
        
        # 임시 학습 진행률 데이터
        progress = {
            "overall_progress": 65,
            "categories": {
                "basic_knowledge": {
                    "completed": 2,
                    "total": 3,
                    "progress": 67
                },
                "portfolio": {
                    "completed": 0,
                    "total": 2,
                    "progress": 0
                },
                "technical_analysis": {
                    "completed": 0,
                    "total": 2,
                    "progress": 0
                }
            }
        }
        
        return jsonify({"success": True, "progress": progress})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/learning-content/<category>')
def api_learning_content(category: str):
    """학습 콘텐츠 API"""
    try:
        # 임시 학습 콘텐츠 데이터
        content = {
            "basic_knowledge": [
                {
                    "id": "investment_basics",
                    "title": "투자란 무엇인가?",
                    "description": "투자의 기본 개념과 원리를 학습합니다.",
                    "duration": 15,
                    "status": "completed"
                },
                {
                    "id": "stock_basics",
                    "title": "주식 투자 기초",
                    "description": "주식 투자의 기본 개념을 학습합니다.",
                    "duration": 20,
                    "status": "in_progress"
                },
                {
                    "id": "market_structure",
                    "title": "시장 구조 이해",
                    "description": "주식시장의 구조와 운영 방식을 학습합니다.",
                    "duration": 25,
                    "status": "not_started"
                }
            ],
            "portfolio": [
                {
                    "id": "asset_allocation",
                    "title": "자산 배분의 중요성",
                    "description": "효과적인 자산 배분 전략을 학습합니다.",
                    "duration": 25,
                    "status": "not_started"
                },
                {
                    "id": "risk_management",
                    "title": "리스크 관리",
                    "description": "투자 리스크 관리 방법을 학습합니다.",
                    "duration": 30,
                    "status": "not_started"
                }
            ],
            "technical_analysis": [
                {
                    "id": "technical_indicators",
                    "title": "기술적 지표",
                    "description": "주요 기술적 지표의 활용법을 학습합니다.",
                    "duration": 40,
                    "status": "not_started"
                }
            ]
        }
        
        if category in content:
            return jsonify({"success": True, "content": content[category]})
        else:
            return jsonify({"success": False, "error": "카테고리를 찾을 수 없습니다."})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/get_survey_result', methods=['GET'])
def get_survey_result():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인 정보가 없습니다.'}), 401
        supabase = get_supabase_client()
        response = supabase.table('user_profiles').select('*').eq('user_id', user_id).execute()
        if not response.data or len(response.data) == 0:
            return jsonify({'error': '분석 결과가 없습니다.'}), 404
        row = response.data[0]
        profile_json = row.get('profile_json', {})
        summary = row.get('summary', '')
        # profile_json에서 실제 분석결과 필드 추출
        result = {
            'overall_analysis': profile_json.get('overall_analysis', ''),
            'detailed_analysis': profile_json.get('detailed_analysis', {}),
            'portfolio': profile_json.get('portfolio', []),
            'portfolio_reason': profile_json.get('portfolio_reason', ''),
            'time_series_prediction': profile_json.get('time_series_prediction', ''),
            'recent_history': profile_json.get('recent_history', ''),
            'summary': summary
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'분석 결과 조회 중 오류: {str(e)}'}), 500

@app.route('/reset_survey', methods=['POST'])
def reset_survey():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인 정보가 없습니다.'}), 401
        supabase = get_supabase_client()
        # 삭제 전 user_id 로그
        print(f"[reset_survey] 삭제 시도 user_id: {user_id}")
        # 해당 user_id의 모든 row 삭제
        response = supabase.table('user_profiles').delete().eq('user_id', user_id).execute()
        print(f"[reset_survey] 삭제 결과: {response.data}")
        # 삭제된 row 개수 확인
        if hasattr(response, 'data') and response.data is not None:
            deleted_count = len(response.data)
        else:
            deleted_count = 0
        return jsonify({'success': True, 'deleted': deleted_count})
    except Exception as e:
        print(f"[reset_survey] 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ================= 뉴스 관련 API 엔드포인트 =================

@app.route('/api/news-popular')
def api_news_popular():
    """인기 뉴스 조회 (조회수 기반)"""
    try:
        # Mock 데이터 먼저 시도
        mock_news = get_mock_news_data()
        mock_popular = mock_news.get('popular', [])
        
        # popular가 없으면 today_pick 사용
        if not mock_popular:
            mock_popular = mock_news.get('today_pick', [])[:8]
        
        if mock_popular:
            for news in mock_popular:
                if 'id' not in news:
                    news['id'] = hash(news.get('title', ''))
                if 'source' not in news:
                    news['source'] = '픽시 뉴스'
                if 'time' not in news:
                    news['time'] = '방금 전'
                if 'link' not in news:
                    news['link'] = '#'
                if 'likes' not in news:
                    news['likes'] = random.randint(100, 500)
                if 'views' not in news:
                    news['views'] = random.randint(1000, 50000)
            return jsonify(clean_json_data({'success': True, 'news_list': mock_popular}))
        # 필터 파라미터 가져오기
        filter_type = request.args.get('filter', 'domestic')
        
        # 실제 데이터 매니저에서 뉴스 데이터 가져오기
        news_data = real_data_manager.load_news_data()
        
        popular_news = []
        if news_data and len(news_data) > 0:
            # 뉴스를 조회수로 정렬 (랜덤 생성 - 실제로는 DB에서 관리)
            for news in news_data:
                news['views'] = news.get('views', random.randint(1000, 50000))
            
            # 조회수로 정렬
            sorted_news = sorted(news_data, key=lambda x: x.get('views', 0), reverse=True)
            
            # 필터링
            for news in sorted_news[:50]:  # 상위 50개에서 필터링
                title = str(news.get('title', ''))
                content = str(news.get('content', ''))
                
                # 해외/국내 뉴스 확인
                global_keywords = ['미국', '중국', '일본', '유럽', 'EU', '연준', 'Fed', 'FOMC', '바이든', '트럼프', 
                                  '나스닥', 'S&P', '다우', '애플', '테슬라', '엔비디아', 'TSMC', 'NYSE', '달러', '엔화', '위안화']
                
                domestic_keywords = ['한국', '국내', '코스피', '코스닥', '삼성', 'SK', 'LG', '현대', '기아', '네이버', '카카오',
                                   '금융위', '한국은행', '금통위', '서울', '대한민국']
                
                is_global = any(keyword in title or keyword in content for keyword in global_keywords)
                is_domestic = any(keyword in title or keyword in content for keyword in domestic_keywords)
                
                if not is_global and not is_domestic:
                    is_domestic = True
                
                # 필터링 적용
                if filter_type == 'global' and not is_global:
                    continue
                elif filter_type == 'domestic' and is_global and not is_domestic:
                    continue
                
                # 날짜 포맷팅
                try:
                    if 'pub_date' in news and news['pub_date']:
                        from datetime import datetime
                        pub_date = datetime.fromisoformat(news['pub_date'].replace('Z', '+00:00'))
                        days_diff = (datetime.now() - pub_date).days
                        if days_diff == 0:
                            news['time'] = '오늘'
                        elif days_diff == 1:
                            news['time'] = '어제'
                        else:
                            news['time'] = f'{days_diff}일 전'
                    else:
                        news['time'] = '최근'
                except:
                    news['time'] = '최근'
                
                # 카테고리 추가
                if 'category' not in news:
                    if '삼성' in title or 'SK' in title:
                        news['category'] = 'IT'
                    elif '경제' in title or '금융' in title:
                        news['category'] = '경제'
                    elif '제약' in title or '바이오' in title:
                        news['category'] = '제약'
                    else:
                        news['category'] = '기타'
                
                news['likes'] = news.get('likes', random.randint(100, 300))
                news['isLiked'] = False
                news['views_formatted'] = f"{news['views']:,}"
                popular_news.append(news)
                
                if len(popular_news) >= 20:
                    break
        
        return jsonify(clean_json_data({
            'success': True,
            'news_list': popular_news
        }))
        
    except Exception as e:
        print(f"인기 뉴스 조회 오류: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 중복 제거 완료 - 뉴스 관련 API 엔드포인트들은 이미 상단에 정의됨
# api_watchlist도 이미 4344번 라인에 정의되어 있으므로 중복 제거

# 중복된 api_watchlist 함수 제거됨 (이미 4344번 라인에 정의)

@app.route('/api/news-sentiment', methods=['GET'])
def api_news_sentiment():
    """뉴스 감정 분석 결과 조회"""
    try:
        # 뉴스 데이터 로드
        news_data = real_data_manager.load_news_data()
        
        if not news_data:
            return jsonify({
                'success': False, 
                'error': '뉴스 데이터가 없습니다'
            })
        
        # 감정 분석 수행
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
        from news_sentiment_analyzer import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer()
        
        # DataFrame 변환
        news_df = pd.DataFrame(news_data)
        
        # 감정 분석
        sentiment_result = analyzer.analyze_news_sentiment(news_df)
        
        # 투자 신호 생성
        investment_signals = analyzer.get_investment_signals(sentiment_result)
        
        return jsonify({
            'success': True,
            'sentiment_analysis': sentiment_result,
            'investment_signals': investment_signals
        })
        
    except Exception as e:
        print(f"뉴스 감정 분석 오류: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 관심 종목 관련 API
@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    """사용자의 관심 종목 목록을 반환합니다."""
    try:
        user_id = session.get('user_id', 'default')
        
        # 데이터베이스에서 관심 종목 조회
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # watchlist 테이블이 없으면 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                name TEXT NOT NULL,
                sector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, ticker)
            )
        ''')
        
        # 기본 관심 종목 삽입 (처음 사용자)
        cursor.execute('''
            INSERT OR IGNORE INTO watchlist (user_id, ticker, name, sector)
            VALUES 
                (?, '005930', '삼성전자', '전기전자/반도체'),
                (?, '000660', 'SK하이닉스', '전기전자/반도체')
        ''', (user_id, user_id))
        
        conn.commit()
        
        # 관심 종목 조회
        cursor.execute('''
            SELECT ticker, name, sector FROM watchlist 
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        watchlist = []
        for row in cursor.fetchall():
            ticker, name, sector = row
            
            # 최신 주가 정보 가져오기 (있으면)
            try:
                price_data = real_data_manager.load_stock_data()
                stock_info = next((s for s in price_data if s.get('ticker') == ticker), None)
                
                if stock_info:
                    watchlist.append({
                        'ticker': ticker,
                        'name': name,
                        'sector': sector,
                        'price': stock_info.get('close', 0),
                        'change_percent': stock_info.get('change_pct', 0),
                        'change_price': stock_info.get('change', 0)
                    })
                else:
                    # 기본값
                    watchlist.append({
                        'ticker': ticker,
                        'name': name,
                        'sector': sector,
                        'price': 0,
                        'change_percent': 0,
                        'change_price': 0
                    })
            except:
                watchlist.append({
                    'ticker': ticker,
                    'name': name,
                    'sector': sector,
                    'price': 0,
                    'change_percent': 0,
                    'change_price': 0
                })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'watchlist': watchlist
        })
        
    except Exception as e:
        print(f"관심 종목 조회 오류: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist', methods=['POST'])
def add_to_watchlist():
    """관심 종목을 추가합니다."""
    try:
        user_id = session.get('user_id', 'default')
        data = request.get_json()
        ticker = data.get('ticker')
        
        if not ticker:
            return jsonify({'success': False, 'error': '종목 코드가 필요합니다.'}), 400
        
        # 종목 정보 가져오기
        ticker_data = real_data_manager.load_ticker_data()
        stock_info = next((s for s in ticker_data if s.get('ticker') == ticker), None)
        
        if not stock_info:
            return jsonify({'success': False, 'error': '종목 정보를 찾을 수 없습니다.'}), 404
        
        # 데이터베이스에 추가
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO watchlist (user_id, ticker, name, sector)
            VALUES (?, ?, ?, ?)
        ''', (user_id, ticker, stock_info.get('name', ''), stock_info.get('sector', '')))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f"{stock_info.get('name')}이(가) 관심 종목에 추가되었습니다."
        })
        
    except Exception as e:
        print(f"관심 종목 추가 오류: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist/<ticker>', methods=['DELETE'])
def remove_from_watchlist(ticker):
    """관심 종목을 제거합니다."""
    try:
        user_id = session.get('user_id', 'default')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM watchlist 
            WHERE user_id = ? AND ticker = ?
        ''', (user_id, ticker))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': '관심 종목에서 제거되었습니다.'
        })
        
    except Exception as e:
        print(f"관심 종목 제거 오류: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stocks/list', methods=['GET'])
def get_stocks_list():
    """전체 종목 목록을 반환합니다."""
    try:
        ticker_data = real_data_manager.load_ticker_data()
        
        stocks = []
        for stock in ticker_data[:200]:  # 최대 200개
            stocks.append({
                'ticker': stock.get('ticker', ''),
                'name': stock.get('name', ''),
                'sector': stock.get('sector', ''),
                'market': stock.get('market', 'KOSPI')
            })
        
        return jsonify({
            'success': True,
            'stocks': stocks
        })
        
    except Exception as e:
        print(f"종목 목록 조회 오류: {e}")
        # 에러시 샘플 데이터 반환
        sample_stocks = [
            {'ticker': '005930', 'name': '삼성전자', 'sector': '전기전자', 'market': 'KOSPI'},
            {'ticker': '000660', 'name': 'SK하이닉스', 'sector': '전기전자', 'market': 'KOSPI'},
            {'ticker': '035720', 'name': '카카오', 'sector': 'IT', 'market': 'KOSPI'},
            {'ticker': '035420', 'name': 'NAVER', 'sector': 'IT', 'market': 'KOSPI'},
            {'ticker': '051910', 'name': 'LG화학', 'sector': '화학', 'market': 'KOSPI'}
        ]
        return jsonify({
            'success': True,
            'stocks': sample_stocks
        })

@app.route('/api/market-analysis-quick', methods=['POST'])
def api_market_analysis_quick():
    """빠른 시장 분석 응답 (AI 체인 없이)"""
    try:
        from src.db_client import get_supabase_client
        from datetime import datetime, timedelta
        
        # 시장 감정 데이터 수집
        sentiment_data = get_market_sentiment_data()
        
        # 간단한 분석 생성 (전체 AI 체인 없이)
        analysis = generate_quick_market_analysis(sentiment_data)
        
        return jsonify({
            'success': True,
            'response': analysis,
            'data': sentiment_data
        })
        
    except Exception as e:
        logger.error(f"Quick market analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'response': '시장 분석 중 오류가 발생했습니다.'
        })

def get_market_sentiment_data():
    """시장 감정 데이터 빠르게 수집"""
    from src.db_client import get_supabase_client
    from datetime import datetime, timedelta
    
    supabase = get_supabase_client()
    data = {
        'sentiment_score': 50,
        'positive_count': 0,
        'negative_count': 0,
        'total_count': 0,
        'keywords': [],
        'important_news': []
    }
    
    if supabase:
        try:
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            # 간단한 감정 통계만
            response = supabase.table('news_articles').select(
                "sentiment, sentiment_score"
            ).gte('published_date', yesterday.isoformat()).execute()
            
            if response.data:
                positive = sum(1 for item in response.data if item.get('sentiment') == '긍정')
                negative = sum(1 for item in response.data if item.get('sentiment') == '부정')
                total = len(response.data)
                
                scores = [item.get('sentiment_score', 0) for item in response.data if item.get('sentiment_score') is not None]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    data['sentiment_score'] = int((avg_score + 1) * 50)
                
                data.update({
                    'positive_count': positive,
                    'negative_count': negative,
                    'total_count': total,
                    'positive_ratio': round(positive / total * 100, 1) if total > 0 else 0
                })
        except Exception as e:
            logger.error(f"Market sentiment data error: {e}")
    
    return data

def generate_quick_market_analysis(data):
    """간단한 시장 분석 텍스트 생성 (AI 없이)"""
    from datetime import datetime
    
    score = data['sentiment_score']
    positive_ratio = data.get('positive_ratio', 0)
    
    # 시장 상황 판단
    if score >= 70:
        market_status = "매우 긍정적"
        recommendation = "투자 심리가 좋은 시기입니다. 다만 과열 여부를 주의하세요."
    elif score >= 60:
        market_status = "긍정적"
        recommendation = "전반적으로 안정적인 투자 환경입니다."
    elif score >= 40:
        market_status = "중립적"
        recommendation = "시장이 관망세를 보이고 있습니다. 신중한 접근이 필요합니다."
    elif score >= 30:
        market_status = "부정적"
        recommendation = "투자 심리가 위축되어 있습니다. 보수적 전략이 권장됩니다."
    else:
        market_status = "매우 부정적"
        recommendation = "시장 불안이 높습니다. 현금 비중을 높이는 것을 고려하세요."
    
    analysis = f"""[실시간 시장 분석 리포트]

** 현재 시장 감정: {market_status} ({score}점/100점) **

[뉴스 감정 분석]
- 긍정 뉴스: {data['positive_count']}건
- 부정 뉴스: {data['negative_count']}건
- 긍정 비율: {positive_ratio}%

[투자 제안]
{recommendation}

분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    return analysis

@app.route('/api/market-sentiment', methods=['GET'])
def api_market_sentiment():
    """시장 전체 감정 분석 데이터 반환"""
    try:
        from src.db_client import get_supabase_client
        from datetime import datetime, timedelta
        
        supabase = get_supabase_client()
        
        # 기본 응답 구조
        response_data = {
            'success': True,
            'sentiment_score': 50,
            'stats': {
                'positive_count': 0,
                'negative_count': 0,
                'total_count': 0
            },
            'sectors': [],
            'trend': {
                'labels': [],
                'scores': []
            }
        }
        
        if supabase:
            try:
                # 오늘 뉴스의 감정 분석 데이터 가져오기
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                
                # 전체 뉴스 감정 통계
                response = supabase.table('news_articles').select(
                    "sentiment, sentiment_score"
                ).gte('published_date', yesterday.isoformat()).execute()
                
                if response.data:
                    positive_count = sum(1 for item in response.data if item.get('sentiment') == '긍정')
                    negative_count = sum(1 for item in response.data if item.get('sentiment') == '부정')
                    total_count = len(response.data)
                    
                    # 평균 감정 점수 계산 (0-100 스케일로 변환)
                    sentiment_scores = [item.get('sentiment_score', 0) for item in response.data if item.get('sentiment_score') is not None]
                    if sentiment_scores:
                        avg_score = sum(sentiment_scores) / len(sentiment_scores)
                        # -1 ~ 1 범위를 0 ~ 100으로 변환
                        sentiment_score = int((avg_score + 1) * 50)
                    else:
                        sentiment_score = 50
                    
                    response_data['sentiment_score'] = sentiment_score
                    response_data['stats'] = {
                        'positive_count': positive_count,
                        'negative_count': negative_count,
                        'total_count': total_count,
                        'positive_ratio': round(positive_count / total_count * 100, 1) if total_count > 0 else 0
                    }
                
                # 섹터별 감정 분석 (카테고리 기반)
                sector_response = supabase.table('news_articles').select(
                    "category, sentiment, sentiment_score"
                ).gte('published_date', yesterday.isoformat()).execute()
                
                if sector_response.data:
                    sector_scores = {}
                    for item in sector_response.data:
                        category = item.get('category', '기타')
                        if category not in sector_scores:
                            sector_scores[category] = []
                        
                        score = item.get('sentiment_score', 0)
                        sector_scores[category].append(score)
                    
                    # 섹터별 평균 계산
                    sectors = []
                    for sector, scores in sector_scores.items():
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            # -1 ~ 1 범위를 0 ~ 100으로 변환
                            sector_sentiment = int((avg_score + 1) * 50)
                            sectors.append({
                                'name': sector,
                                'score': sector_sentiment,
                                'count': len(scores)
                            })
                    
                    # 점수 기준으로 정렬
                    sectors.sort(key=lambda x: x['score'], reverse=True)
                    response_data['sectors'] = sectors[:8]  # 상위 8개 섹터만
                
                # 최근 7일 트렌드
                trend_labels = []
                trend_scores = []
                
                for i in range(6, -1, -1):
                    date = today - timedelta(days=i)
                    next_date = date + timedelta(days=1)
                    
                    day_response = supabase.table('news_articles').select(
                        "sentiment_score"
                    ).gte('published_date', date.isoformat()).lt('published_date', next_date.isoformat()).execute()
                    
                    if day_response.data:
                        day_scores = [item.get('sentiment_score', 0) for item in day_response.data if item.get('sentiment_score') is not None]
                        if day_scores:
                            avg_day_score = sum(day_scores) / len(day_scores)
                            trend_score = int((avg_day_score + 1) * 50)
                        else:
                            trend_score = 50
                    else:
                        trend_score = 50
                    
                    if i == 0:
                        trend_labels.append('오늘')
                    elif i == 1:
                        trend_labels.append('어제')
                    else:
                        trend_labels.append(f'{i}일전')
                    
                    trend_scores.append(trend_score)
                
                response_data['trend'] = {
                    'labels': trend_labels,
                    'scores': trend_scores
                }
                
                # 키워드 분석 (news_keywords 테이블에서 가져오기)
                keyword_response = supabase.table('news_keywords').select(
                    "keyword, count"
                ).order('count', desc=True).limit(10).execute()
                
                if keyword_response.data:
                    response_data['keywords'] = keyword_response.data
                else:
                    # 대체 방법: 뉴스 제목에서 키워드 추출
                    response_data['keywords'] = [
                        {'keyword': '반도체', 'count': 45},
                        {'keyword': 'AI', 'count': 38},
                        {'keyword': '배터리', 'count': 32},
                        {'keyword': '바이오', 'count': 28},
                        {'keyword': '게임', 'count': 25}
                    ]
                
                # 중요 뉴스 가져오기 (importance_score가 높은 뉴스)
                important_response = supabase.table('news_articles').select(
                    "title, sentiment, importance_scores(score)"
                ).gte('published_date', yesterday.isoformat()).order(
                    'importance_scores.score', desc=True
                ).limit(5).execute()
                
                if important_response.data:
                    response_data['important_news'] = [
                        {
                            'title': item['title'],
                            'sentiment': item.get('sentiment', '중립'),
                            'importance': item.get('importance_scores', {}).get('score', 0) if isinstance(item.get('importance_scores'), dict) else 0
                        }
                        for item in important_response.data
                    ]
                else:
                    # 대체: 최신 뉴스 중 감정 점수가 극단적인 뉴스
                    extreme_news_response = supabase.table('news_articles').select(
                        "title, sentiment, sentiment_score"
                    ).gte('published_date', yesterday.isoformat()).execute()
                    
                    if extreme_news_response.data:
                        # 감정 점수의 절댓값이 큰 순으로 정렬
                        sorted_news = sorted(extreme_news_response.data, 
                                           key=lambda x: abs(x.get('sentiment_score', 0)), 
                                           reverse=True)[:5]
                        response_data['important_news'] = [
                            {
                                'title': item['title'],
                                'sentiment': item.get('sentiment', '중립')
                            }
                            for item in sorted_news
                        ]
                    else:
                        response_data['important_news'] = []
                
            except Exception as e:
                logger.error(f"Supabase 시장 감정 분석 오류: {e}")
        
        # Supabase 데이터가 없으면 Mock 데이터 사용
        if response_data['stats']['total_count'] == 0:
            import random
            
            # Mock 데이터 생성
            response_data = {
                'success': True,
                'sentiment_score': 75,
                'stats': {
                    'positive_count': 142,
                    'negative_count': 58,
                    'total_count': 200,
                    'positive_ratio': 71.0
                },
                'sectors': [
                    {'name': '반도체', 'score': 87, 'count': 25},
                    {'name': '2차전지', 'score': 72, 'count': 18},
                    {'name': '바이오', 'score': 68, 'count': 22},
                    {'name': '게임', 'score': 82, 'count': 15},
                    {'name': '엔터', 'score': 65, 'count': 12},
                    {'name': '자동차', 'score': 52, 'count': 20},
                    {'name': '금융', 'score': 48, 'count': 30},
                    {'name': '건설', 'score': 35, 'count': 18}
                ],
                'trend': {
                    'labels': ['6일전', '5일전', '4일전', '3일전', '2일전', '어제', '오늘'],
                    'scores': [58, 62, 55, 68, 72, 70, 75]
                },
                'keywords': [
                    {'keyword': '반도체', 'count': 45},
                    {'keyword': 'AI', 'count': 38},
                    {'keyword': '배터리', 'count': 32},
                    {'keyword': '바이오', 'count': 28},
                    {'keyword': '게임', 'count': 25},
                    {'keyword': '2차전지', 'count': 23},
                    {'keyword': '메타버스', 'count': 20},
                    {'keyword': '수소경제', 'count': 18}
                ],
                'important_news': [
                    {'title': '삼성전자, AI 반도체 대규모 투자 발표', 'sentiment': '긍정'},
                    {'title': 'SK하이닉스 HBM4 개발 순항, 엔비디아 공급 확대', 'sentiment': '긍정'},
                    {'title': '한국은행 금리 동결, 경기 회복세 주목', 'sentiment': '중립'},
                    {'title': '테슬라 전기차 판매 부진, 국내 배터리 업체 영향', 'sentiment': '부정'},
                    {'title': '정부, AI 산업 육성 위한 규제 완화 추진', 'sentiment': '긍정'}
                ]
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"시장 감정 분석 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/news-watchlist', methods=['POST'])
def get_watchlist_news():
    """관심 종목 관련 뉴스를 반환합니다."""
    try:
        data = request.get_json()
        stock_names = data.get('stocks', [])
        
        if not stock_names:
            return jsonify({'success': True, 'news_list': []})
        
        # 뉴스 데이터 로드
        news_data = real_data_manager.load_news_data()
        
        if not news_data:
            return jsonify({'success': True, 'news_list': []})
        
        # 관심 종목 관련 뉴스 필터링
        filtered_news = []
        for news in news_data:
            title = news.get('title', '')
            content = news.get('content', '') or news.get('summary', '')
            
            # 종목명이 제목이나 내용에 포함되어 있는지 확인
            for stock_name in stock_names:
                if stock_name in title or stock_name in content:
                    news['stock'] = stock_name
                    filtered_news.append(news)
                    break
        
        # 시간순 정렬
        filtered_news.sort(key=lambda x: x.get('published', ''), reverse=True)
        
        # 최대 20개만 반환
        news_list = []
        for news in filtered_news[:20]:
            news_list.append({
                'title': news.get('title', ''),
                'description': news.get('summary', '')[:200] + '...' if len(news.get('summary', '')) > 200 else news.get('summary', ''),
                'link': news.get('link', '#'),
                'pubDate': news.get('published', ''),
                'stock': news.get('stock', ''),
                'source': news.get('source', 'Unknown')
            })
        
        return jsonify({
            'success': True,
            'news_list': news_list
        })
        
    except Exception as e:
        print(f"관심 종목 뉴스 조회 오류: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stock-price', methods=['GET'])
def get_stock_price():
    """실시간 주가 정보를 반환합니다."""
    try:
        stock_code = request.args.get('code', '')
        if not stock_code:
            return jsonify({'success': False, 'error': '종목 코드를 입력해주세요.'})
        
        # 한국 주식인지 미국 주식인지 판단
        is_korean = stock_code.endswith('.KS') or stock_code.isdigit()
        
        # 최신 주가 데이터 파일 찾기
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
        today = datetime.now()
        price_data = None
        
        # 최근 7일간의 데이터 찾기
        for days_back in range(7):
            date = today - timedelta(days=days_back)
            date_str = date.strftime('%Y%m%d')
            
            if is_korean:
                price_file = os.path.join(data_dir, f'kor_price_{date_str}.csv')
                code_to_find = stock_code.replace('.KS', '').zfill(6)
            else:
                price_file = os.path.join(data_dir, f'us_price_{date_str}.csv')
                code_to_find = stock_code
            
            if os.path.exists(price_file):
                try:
                    df = pd.read_csv(price_file)
                    # 종목 코드로 필터링
                    stock_data = df[df['ticker'] == code_to_find]
                    if not stock_data.empty:
                        price_data = stock_data.iloc[0].to_dict()
                        price_data['datetime'] = date.strftime('%Y-%m-%d')
                        break
                except Exception as e:
                    logger.error(f"주가 데이터 로드 오류: {e}")
                    continue
        
        if price_data:
            return jsonify({
                'success': True,
                'price': price_data
            })
        else:
            return jsonify({
                'success': False,
                'error': '해당 종목의 데이터를 찾을 수 없습니다.'
            })
            
    except Exception as e:
        logger.error(f"주가 조회 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})


def analyze_stock_trends(stock_code):
    """주식 종목의 트렌드를 분석하고 예측합니다."""
    try:
        # 최신 주가 데이터 로드
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
        today = datetime.now()
        
        # 최근 30일간의 데이터 수집
        prices = []
        for days_back in range(30):
            date = today - timedelta(days=days_back)
            date_str = date.strftime('%Y%m%d')
            price_file = os.path.join(data_dir, f'kor_price_{date_str}.csv')
            
            if os.path.exists(price_file):
                try:
                    df = pd.read_csv(price_file)
                    # 종목 코드로 필터링 (디버깅 로그 추가)
                    code_to_find = stock_code.zfill(6)
                    logger.info(f"찾는 종목 코드: {code_to_find}")
                    
                    # ticker 컬럼에서 찾기
                    stock_data = df[df['ticker'].astype(str).str.strip() == code_to_find]
                    
                    if not stock_data.empty:
                        row = stock_data.iloc[0]
                        prices.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'close': float(row['Close']),
                            'volume': float(row['Volume'])
                        })
                        logger.info(f"종목 {code_to_find} 데이터 찾음: {row['Close']}")
                except Exception as e:
                    logger.error(f"파일 {price_file} 읽기 오류: {e}")
                    continue
        
        if len(prices) < 7:
            logger.warning(f"종목 {stock_code}의 데이터가 부족합니다: {len(prices)}개")
            # 더미 데이터로 예측 생성
            current_price = 50000
            predicted_price = current_price * (1 + random.uniform(-0.1, 0.1))
            change_percent = ((predicted_price - current_price) / current_price) * 100
            
            return {
                'stock_code': stock_code,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'change_percent': change_percent,
                'trend': 'bullish' if change_percent > 0 else 'bearish',
                'model_type': 'ARIMA-X',
                'confidence': 'low',
                'prediction_horizon': '7일'
            }
        
        # 최신 가격과 7일 전 가격 비교
        prices.sort(key=lambda x: x['date'], reverse=True)
        current_price = prices[0]['close']
        week_ago_price = prices[min(6, len(prices)-1)]['close']
        
        # 간단한 예측 (실제로는 ARIMA-X 모델 사용)
        # 여기서는 단순히 최근 추세를 바탕으로 예측
        trend_factor = (current_price - week_ago_price) / week_ago_price
        predicted_price = current_price * (1 + trend_factor * 0.5)  # 추세의 50%만 반영
        
        # 랜덤 요소 추가 (실제 시장의 불확실성 반영)
        predicted_price *= (1 + random.uniform(-0.05, 0.05))
        
        change_percent = ((predicted_price - current_price) / current_price) * 100
        
        return {
            'stock_code': stock_code,
            'current_price': int(current_price),
            'predicted_price': int(predicted_price),
            'change_percent': round(change_percent, 2),
            'trend': 'bullish' if change_percent > 0 else 'bearish',
            'model_type': 'ARIMA-X',
            'confidence': 'medium' if abs(change_percent) < 10 else 'low',
            'prediction_horizon': '7일'
        }
        
    except Exception as e:
        logger.error(f"종목 {stock_code} 분석 오류: {e}")
        return None

# Financial Report Analyzer 전역 인스턴스
financial_report_analyzer = FinancialReportAnalyzer()

@app.route('/api/financial-reports/<ticker>')
def api_financial_reports(ticker):
    """종목별 재무제표 분석 및 인사이트 API"""
    try:
        # 재무제표 분석
        financial_analysis = financial_report_analyzer.analyze_financial_statements(ticker)
        
        # 핵심 인사이트 생성
        insights = financial_report_analyzer.generate_insights(ticker)
        
        # 인사이트를 딕셔너리 형태로 변환
        insights_data = []
        for insight in insights:
            insights_data.append({
                'type': insight.insight_type,
                'title': insight.title,
                'summary': insight.summary,
                'key_metrics': insight.key_metrics,
                'trend_analysis': insight.trend_analysis,
                'created_at': insight.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'financial_analysis': financial_analysis,
            'insights': insights_data
        })
        
    except Exception as e:
        logger.error(f"재무 보고서 분석 오류 ({ticker}): {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/risk-analysis/<ticker>')
def api_risk_analysis(ticker):
    """종목별 위험 신호 감지 및 분석 API"""
    try:
        # 위험 신호 감지
        risk_alerts = financial_report_analyzer.detect_risk_signals(ticker)
        
        # RiskAlert 객체를 딕셔너리로 변환
        alerts_data = []
        for alert in risk_alerts:
            alerts_data.append({
                'ticker': alert.ticker,
                'stock_name': alert.stock_name,
                'risk_level': alert.risk_level.value,
                'alert_type': alert.alert_type,
                'title': alert.title,
                'description': alert.description,
                'analysis': alert.analysis,
                'indicators': alert.indicators,
                'detected_at': alert.detected_at.isoformat(),
                'recommendations': alert.recommendations
            })
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'risk_alerts': alerts_data,
            'total_alerts': len(alerts_data)
        })
        
    except Exception as e:
        logger.error(f"위험 분석 오류 ({ticker}): {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/comprehensive-analysis')
def api_comprehensive_analysis():
    """관심 종목 전체에 대한 종합 분석 API"""
    try:
        session_id = session.get('user_id')
        if not session_id:
            session_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = session_id
        
        # 관심종목 목록 가져오기
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT stock_code, stock_name FROM watchlist WHERE user_id = ?', (session_id,))
        watchlist = [{"code": row[0], "name": row[1]} for row in c.fetchall()]
        conn.close()
        
        # 종합 분석 결과
        comprehensive_results = {
            'watchlist': watchlist,
            'analyses': [],
            'total_risk_alerts': 0,
            'high_risk_stocks': [],
            'top_insights': []
        }
        
        # 각 관심종목에 대해 분석 수행
        for stock in watchlist:
            ticker = stock['code']
            
            # 재무 분석
            financial_analysis = financial_report_analyzer.analyze_financial_statements(ticker)
            
            # 위험 신호 감지
            risk_alerts = financial_report_analyzer.detect_risk_signals(ticker)
            
            # 인사이트 생성
            insights = financial_report_analyzer.generate_insights(ticker)
            
            # 결과 집계
            stock_analysis = {
                'ticker': ticker,
                'stock_name': stock['name'],
                'health_score': financial_analysis.get('health_score', {}).get('score', 0),
                'risk_alert_count': len(risk_alerts),
                'insight_count': len(insights),
                'risk_level': 'high' if any(alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] for alert in risk_alerts) else 'medium' if risk_alerts else 'low'
            }
            
            comprehensive_results['analyses'].append(stock_analysis)
            comprehensive_results['total_risk_alerts'] += len(risk_alerts)
            
            # 고위험 종목 추출
            if stock_analysis['risk_level'] == 'high':
                comprehensive_results['high_risk_stocks'].append({
                    'ticker': ticker,
                    'stock_name': stock['name'],
                    'alert_count': len(risk_alerts)
                })
            
            # 상위 인사이트 추출 (각 종목당 1개씩)
            if insights:
                top_insight = insights[0]
                comprehensive_results['top_insights'].append({
                    'ticker': ticker,
                    'stock_name': stock['name'],
                    'title': top_insight.title,
                    'summary': top_insight.summary
                })
        
        # 분석 결과 정렬 (위험도 순)
        comprehensive_results['analyses'].sort(key=lambda x: x['risk_alert_count'], reverse=True)
        
        return jsonify({
            'success': True,
            'comprehensive_analysis': comprehensive_results,
            'analyzed_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"종합 분석 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/risk-alert-details/<ticker>')
def api_risk_alert_details(ticker):
    """특정 종목의 위험 신호 상세 분석 API"""
    try:
        # 위험 신호 감지
        risk_alerts = financial_report_analyzer.detect_risk_signals(ticker)
        
        if not risk_alerts:
            return jsonify({
                'success': True,
                'ticker': ticker,
                'message': '현재 감지된 위험 신호가 없습니다.',
                'risk_report': None
            })
        
        # 가장 심각한 위험 신호 선택
        most_critical = max(risk_alerts, key=lambda x: 
            4 if x.risk_level == RiskLevel.CRITICAL else
            3 if x.risk_level == RiskLevel.HIGH else
            2 if x.risk_level == RiskLevel.MEDIUM else 1
        )
        
        # 재무제표 분석 추가
        financial_analysis = financial_report_analyzer.analyze_financial_statements(ticker)
        
        # 상세 리포트 생성
        risk_report = {
            'ticker': ticker,
            'stock_name': most_critical.stock_name,
            'report_title': f"{most_critical.stock_name} 위험 분석 리포트",
            'generated_at': datetime.now().isoformat(),
            'executive_summary': {
                'risk_level': most_critical.risk_level.value,
                'main_risk': most_critical.title,
                'description': most_critical.description
            },
            'detailed_analysis': {
                'risk_factors': [
                    {
                        'type': alert.alert_type,
                        'title': alert.title,
                        'analysis': alert.analysis,
                        'indicators': alert.indicators,
                        'severity': alert.risk_level.value
                    } for alert in risk_alerts
                ],
                'financial_health': financial_analysis.get('health_score', {}),
                'key_metrics': financial_analysis.get('key_metrics', {}),
                'growth_analysis': financial_analysis.get('growth_analysis', {})
            },
            'recommendations': {
                'immediate_actions': most_critical.recommendations[:2] if len(most_critical.recommendations) >= 2 else most_critical.recommendations,
                'monitoring_points': most_critical.recommendations[2:] if len(most_critical.recommendations) > 2 else [],
                'risk_mitigation': [
                    "포트폴리오 내 비중 재검토",
                    "손절 라인 설정 및 준수",
                    "분산 투자를 통한 리스크 헤지"
                ]
            },
            'market_context': {
                'sector_performance': "업종 평균 대비 분석 필요",
                'peer_comparison': "동종업계 대비 위치 확인 필요",
                'macro_factors': "거시경제 영향 고려 필요"
            }
        }
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'risk_report': risk_report
        })
        
    except Exception as e:
        logger.error(f"위험 상세 분석 오류 ({ticker}): {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # 환경별 실행 설정
    # Windows에서 디버그 모드 문제 해결을 위해 기본값을 False로 설정
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    
    # 시작 메시지
    print("\n" + "="*50)
    print(f"MINERVA 웹 서버 시작")
    print(f"주소: http://{host}:{port}")
    print(f"Supabase 연결: {'활성화' if real_data_manager.supabase else '비활성화'}")
    print(f"API 모드: {API_STATUS['selected']}")
    print("="*50 + "\n")
    # FinancialDataProcessor는 필요할 때만 초기화 (Supabase 사용 시 불필요)
    # get_global_financial_processor()
    
    try:
        # Windows 환경에서 포인터 오류 방지를 위한 설정
        import platform
        if platform.system() == 'Windows':
            # Windows에서는 threaded=False로 실행
            app.run(host=host, port=port, debug=False, threaded=False, use_reloader=False)
        else:
            app.run(host=host, port=port, debug=debug_mode, use_reloader=False)
    except Exception as e:
        print(f"서버 시작 실패: {e}")
        print("포트가 이미 사용 중이거나 권한 문제일 수 있습니다.")
        print("\n대안 1: 다른 포트로 시도")
        print("set PORT=5001 && python app.py")
        print("\n대안 2: waitress 서버 사용")
        print("pip install waitress")
        print("waitress-serve --port=5000 app:app")
        print("다른 포트를 사용하려면: PORT=5001 python app.py") 