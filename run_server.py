"""
Pixie 투자챗봇 서버 실행 스크립트
포인터 오류를 해결하기 위한 간단한 실행 방식
"""
import os
import sys
from flask import Flask, render_template, request, jsonify, session
import uuid
import sqlite3
from datetime import datetime

# 현재 디렉토리를 파이썬 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Flask 앱 생성
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'pixie-investment-chatbot-secure-key-2024')

# 데이터베이스 경로
DB_PATH = os.path.join(current_dir, 'pixie_investment.db')

def init_db():
    """간단한 데이터베이스 초기화"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # users 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_data TEXT,
            last_login TIMESTAMP,
            email TEXT,
            name TEXT
        )''')
        
        # user_profiles 테이블  
        c.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            survey_data TEXT,
            risk_tolerance INTEGER,
            investment_period INTEGER,
            investment_style TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )''')
        
        # chat_history 테이블
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            ai_response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        conn.commit()
        conn.close()
        print("데이터베이스 초기화 완료")
        return True
    except Exception as e:
        print(f"데이터베이스 초기화 오류: {e}")
        return False

# 라우트 설정
@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('index.html')

@app.route('/survey')
def survey():
    """투자 성향 설문 페이지"""
    return render_template('survey.html')

@app.route('/chatbot')
def chatbot():
    """AI 챗봇 페이지"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chatbot.html')

@app.route('/stock')
def stock():
    """주식 예측 페이지"""
    return render_template('stock.html')

@app.route('/news')
def news():
    """뉴스 분석 페이지"""
    return render_template('news.html')

@app.route('/my-investment')
def my_investment():
    """마이 포트폴리오 페이지"""
    return render_template('my-investment.html')

# 기본 API 엔드포인트들
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """간단한 챗봇 API - 시뮬레이션 모드"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        # 세션 관리
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # 간단한 응답 로직
        if '삼성전자' in user_message:
            response = "삼성전자는 현재 안정적인 성장세를 보이고 있습니다. AI와 반도체 시장의 회복으로 긍정적인 전망을 가지고 있습니다."
        elif '투자' in user_message or '주식' in user_message:
            response = "투자 시에는 분산 투자와 장기적인 관점을 유지하는 것이 중요합니다. 위험 관리를 철저히 하시기 바랍니다."
        elif '포트폴리오' in user_message:
            response = "개인의 투자성향과 위험도에 맞는 포트폴리오 구성이 중요합니다. 설문을 통해 맞춤형 포트폴리오를 추천받으실 수 있습니다."
        else:
            response = f"'{user_message}'에 대해 분석 중입니다. 더 구체적인 질문을 해주시면 더 정확한 답변을 드릴 수 있습니다."
        
        # 채팅 기록 저장
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('INSERT INTO chat_history (session_id, user_message, ai_response) VALUES (?, ?, ?)',
                     (session['session_id'], user_message, response))
            conn.commit()
            conn.close()
        except:
            pass  # 저장 실패 시 무시
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/survey', methods=['POST'])
def api_survey():
    """설문 결과 저장 및 분석"""
    try:
        data = request.get_json()
        
        # 세션 ID 생성
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # 간단한 위험도 계산
        risk_scores = []
        for i in range(1, 11):
            score = int(data.get(f'q{i}', 3))
            risk_scores.append(score)
        
        risk_average = sum(risk_scores) / len(risk_scores) if risk_scores else 3
        
        # 투자 성향 분류
        if risk_average >= 4:
            investment_style = '적극적'
            recommendation = '성장주 중심의 포트폴리오를 추천합니다.'
        elif risk_average <= 2:
            investment_style = '보수적' 
            recommendation = '안정적인 배당주와 채권 중심의 포트폴리오를 추천합니다.'
        else:
            investment_style = '균형적'
            recommendation = '성장주와 안정주를 균형있게 구성한 포트폴리오를 추천합니다.'
        
        # 결과 저장
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''INSERT OR REPLACE INTO user_profiles 
                        (user_id, survey_data, risk_tolerance, investment_style) 
                        VALUES (?, ?, ?, ?)''',
                     (session['session_id'], str(data), risk_average, investment_style))
            conn.commit()
            conn.close()
        except:
            pass  # 저장 실패 시 무시
        
        return jsonify({
            'success': True,
            'profile': {
                'risk_tolerance': risk_average,
                'investment_style': investment_style,
                'recommendation': recommendation
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/news-today', methods=['GET'])
def api_news_today():
    """오늘의 뉴스 - 시뮬레이션"""
    sample_news = [
        {
            'content': 'AI 반도체 시장, 2024년 30% 성장 전망',
            'summary': 'AI 기술 발전으로 반도체 수요 급증, 관련 주식들 상승세',
            'sentiment': '긍정',
            'date': '2024-07-23',
            'url': '#'
        },
        {
            'content': '삼성전자, 2분기 실적 시장 예상치 상회',
            'summary': '반도체 부문 실적 개선으로 주가 상승 요인 제공',
            'sentiment': '긍정', 
            'date': '2024-07-23',
            'url': '#'
        }
    ]
    return jsonify({'success': True, 'news_list': sample_news})

# 기본 설정
if __name__ == '__main__':
    # 데이터베이스 초기화
    init_db()
    
    # 환경 설정
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print("=" * 50)
    print("Pixie 투자챗봇 웹서비스 시작")
    print(f"주소: http://{host}:{port}")
    print(f"모드: {'개발' if debug else '운영'}")
    print(f"UI: Pixie 디자인 시스템")
    print("=" * 50)
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False  # 포인터 오류 방지
        )
    except Exception as e:
        print(f"서버 시작 실패: {e}")
        print("다른 포트로 시도: PORT=5001 python run_server.py")