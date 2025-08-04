"""
애플리케이션 설정
환경 변수 및 공통 설정 관리
"""

import os
from datetime import timedelta

class Config:
    """기본 설정"""
    # Flask 설정
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'minerva_investment_advisor_secure_key_2024')
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    SESSION_COOKIE_SECURE = os.environ.get('FLASK_ENV') == 'production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    TEMPLATES_AUTO_RELOAD = True
    
    # 데이터베이스 설정
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'newsbot.db')
    INVESTMENT_DB_PATH = os.environ.get('INVESTMENT_DB_PATH', 'investment_data.db')
    
    # API 설정
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    CLOVA_API_KEY = os.environ.get('CLOVA_API_KEY')
    CLOVA_API_HOST = os.environ.get('CLOVA_API_HOST', 'https://api.clovastudio.com')
    CLOVA_REQUEST_ID = os.environ.get('CLOVA_REQUEST_ID', '')
    
    # Supabase 설정
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
    
    # 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC_DIR = os.path.join(BASE_DIR, 'src')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    # 파일 경로
    HISTORY_FILE = os.path.join(BASE_DIR, 'web', 'chat_history.json')
    ANALYSIS_FILE = os.path.join(BASE_DIR, 'web', 'analysis_results.json')
    
    # 프롬프트 파일 경로
    PROMPT_FILES = {
        "AI-A": os.path.join(SRC_DIR, "prompt_AI-A.txt"),
        "AI-A2": os.path.join(SRC_DIR, "prompt_AI-A2.txt"),
        "AI-B": os.path.join(SRC_DIR, "prompt_AI-B.txt"),
        "survey-analysis": os.path.join(SRC_DIR, "prompt_survey-analysis.txt"),
        "survey-score": os.path.join(SRC_DIR, "prompt_survey-score.txt")
    }
    
    # API 재시도 설정
    API_MAX_RETRIES = int(os.environ.get('API_MAX_RETRIES', 3))
    API_RETRY_DELAY = int(os.environ.get('API_RETRY_DELAY', 1))
    
    # 기능 플래그
    ENABLE_NEWS_PROCESSOR = os.environ.get('ENABLE_NEWS_PROCESSOR', 'false').lower() == 'true'
    ENABLE_SCHEDULER = os.environ.get('ENABLE_SCHEDULER', 'false').lower() == 'true'
    
    # 로깅 설정
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 캐시 설정
    CACHE_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT', 300))  # 5분
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 60))
    
    @classmethod
    def check_api_availability(cls):
        """사용 가능한 API 확인"""
        api_status = {
            'openai': bool(cls.OPENAI_API_KEY and len(cls.OPENAI_API_KEY.strip()) > 10),
            'clova': bool(cls.CLOVA_API_KEY and len(cls.CLOVA_API_KEY.strip()) > 10),
            'simulation': True
        }
        
        if api_status['openai']:
            api_status['selected'] = 'openai'
        elif api_status['clova']:
            api_status['selected'] = 'clova'
        else:
            api_status['selected'] = 'simulation'
        
        return api_status

class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """프로덕션 환경 설정"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True

class TestingConfig(Config):
    """테스트 환경 설정"""
    TESTING = True
    DATABASE_PATH = ':memory:'
    WTF_CSRF_ENABLED = False

# 환경별 설정 매핑
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """현재 환경에 맞는 설정 반환"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])