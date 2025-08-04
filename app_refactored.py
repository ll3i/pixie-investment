"""
리팩토링된 Flask 애플리케이션
Blueprint를 사용한 모듈화 구조
"""

import os
import sys
from flask import Flask, render_template, session
import uuid
import logging
from datetime import datetime

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)  # web 디렉토리를 경로에 추가
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# 설정 임포트
from config import get_config

# Blueprint 임포트
from blueprints.auth import auth_bp
from blueprints.chat import chat_bp
from blueprints.news import news_bp
from blueprints.stock import stock_bp
from blueprints.learning import learning_bp
from blueprints.alerts import alerts_bp

# 애플리케이션 팩토리
def create_app(config_name=None):
    """Flask 애플리케이션 생성"""
    app = Flask(__name__)
    
    # 설정 로드
    config_name = config_name or os.environ.get('FLASK_ENV', 'development')
    app.config.from_object(get_config())
    
    # 로깅 설정
    setup_logging(app)
    
    # 확장 초기화
    init_extensions(app)
    
    # Blueprint 등록
    register_blueprints(app)
    
    # 에러 핸들러 등록
    register_error_handlers(app)
    
    # 기본 라우트 등록
    register_main_routes(app)
    
    # 데이터베이스 초기화
    with app.app_context():
        init_database()
    
    return app

def setup_logging(app):
    """로깅 설정"""
    log_level = getattr(logging, app.config['LOG_LEVEL'], logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=app.config['LOG_FORMAT'],
        handlers=[
            logging.FileHandler(f'{app.config.get("LOG_FILE_PREFIX", "pixie")}_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    app.logger.setLevel(log_level)
    app.logger.info(f"애플리케이션 시작 - 환경: {app.config.get('ENV', 'development')}")

def init_extensions(app):
    """Flask 확장 초기화"""
    # 필요한 확장들 초기화
    # 예: db.init_app(app), migrate.init_app(app, db) 등
    pass

def register_blueprints(app):
    """Blueprint 등록"""
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(news_bp)
    app.register_blueprint(stock_bp)
    app.register_blueprint(learning_bp)
    app.register_blueprint(alerts_bp)
    
    app.logger.info("Blueprint 등록 완료")

def register_error_handlers(app):
    """에러 핸들러 등록"""
    @app.errorhandler(404)
    def not_found_error(error):
        from flask import request
        app.logger.warning(f"404 오류: {request.url}")
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"500 오류: {error}")
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(Exception)
    def unhandled_exception(error):
        app.logger.error(f"처리되지 않은 예외: {error}", exc_info=True)
        return render_template('errors/500.html'), 500

def register_main_routes(app):
    """메인 라우트 등록"""
    @app.route('/')
    def index():
        """메인 페이지"""
        # 세션 확인 및 생성
        if 'user_id' not in session:
            session['user_id'] = f"user_{str(uuid.uuid4())[:8]}"
            app.logger.info(f"새 사용자 생성: {session['user_id']}")
        
        return render_template('index.html')
    
    @app.route('/dashboard')
    def dashboard():
        """대시보드 페이지"""
        return render_template('dashboard.html')
    
    @app.route('/minerva')
    def minerva():
        """Minerva 메인 페이지"""
        return render_template('minerva.html')
    
    @app.route('/chatbot')
    def chatbot():
        """챗봇 페이지"""
        return render_template('chatbot.html')
    
    @app.route('/news')
    def news():
        """뉴스 페이지"""
        return render_template('news.html')
    
    @app.route('/stock')
    def stock():
        """주식 정보 페이지"""
        return render_template('stock.html')
    
    @app.route('/alerts')
    def alerts():
        """알림 페이지"""
        return render_template('alerts.html')
    
    @app.route('/my-invest')
    def my_invest():
        """MY 투자 페이지"""
        return render_template('my_invest.html')
    
    @app.route('/learning')
    def learning():
        """학습 페이지"""
        return render_template('learning.html')
    
    @app.route('/time_series')
    def time_series():
        """시계열 예측 페이지"""
        return render_template('time_series_prediction.html')
    
    # 컨텍스트 프로세서
    @app.context_processor
    def inject_user():
        """모든 템플릿에 사용자 정보 주입"""
        return {
            'user_id': session.get('user_id'),
            'current_year': datetime.now().year
        }

def init_database():
    """데이터베이스 초기화"""
    from services.database_service import DatabaseService
    db_service = DatabaseService()
    db_service.init_database()
    
    # API 상태 확인
    from config import Config
    api_status = Config.check_api_availability()
    if api_status['selected'] == 'simulation':
        logging.warning("API 키가 설정되지 않아 시뮬레이션 모드로 실행됩니다.")
    else:
        logging.info(f"API 모드: {api_status['selected']}")

# 애플리케이션 인스턴스 생성
app = create_app()

# 개발 서버 실행
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )