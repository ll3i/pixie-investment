"""
데이터베이스 관리 서비스
테이블 생성, 마이그레이션 등
"""

import sqlite3
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, db_path='newsbot.db'):
        self.db_path = db_path
    
    def init_database(self):
        """데이터베이스 초기화 및 필수 테이블 생성"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            
            # 포트폴리오 테이블
            c.execute('''CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT,
                quantity INTEGER,
                avg_price REAL,
                current_price REAL,
                market TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )''')
            
            # 뉴스 테이블
            c.execute('''CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                url TEXT,
                source TEXT,
                category TEXT,
                published_at TIMESTAMP,
                sentiment REAL,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # 학습 진행 상황 테이블
            c.execute('''CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                module_id TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                completed BOOLEAN DEFAULT 0,
                score INTEGER,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )''')
            
            # 기존 테이블 마이그레이션
            self._migrate_tables(c)
            
            conn.commit()
            logger.info("데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            if 'conn' in locals():
                conn.rollback()
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _migrate_tables(self, cursor):
        """기존 테이블에 필요한 컬럼 추가 (마이그레이션)"""
        try:
            # alerts_history 테이블에 is_read 컬럼 확인
            cursor.execute("PRAGMA table_info(alerts_history)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'is_read' not in columns:
                cursor.execute('ALTER TABLE alerts_history ADD COLUMN is_read BOOLEAN DEFAULT 0')
                logger.info("alerts_history 테이블에 is_read 컬럼 추가")
            
            # users 테이블에 추가 컬럼들 확인
            cursor.execute("PRAGMA table_info(users)")
            user_columns = [row[1] for row in cursor.fetchall()]
            
            if 'last_login' not in user_columns:
                cursor.execute('ALTER TABLE users ADD COLUMN last_login TIMESTAMP')
                logger.info("users 테이블에 last_login 컬럼 추가")
                
            if 'email' not in user_columns:
                cursor.execute('ALTER TABLE users ADD COLUMN email TEXT')
                logger.info("users 테이블에 email 컬럼 추가")
                
            if 'name' not in user_columns:
                cursor.execute('ALTER TABLE users ADD COLUMN name TEXT')
                logger.info("users 테이블에 name 컬럼 추가")
                
        except Exception as e:
            logger.error(f"테이블 마이그레이션 중 오류: {e}")
    
    def get_connection(self):
        """데이터베이스 연결 반환"""
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query, params=None):
        """쿼리 실행"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"쿼리 실행 오류: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def fetch_one(self, query, params=None):
        """단일 결과 조회"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            return cursor.fetchone()
            
        except Exception as e:
            logger.error(f"조회 오류: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def fetch_all(self, query, params=None):
        """전체 결과 조회"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            return cursor.fetchall()
            
        except Exception as e:
            logger.error(f"조회 오류: {e}")
            raise
        finally:
            if conn:
                conn.close()