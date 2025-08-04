"""
Vercel 배포용 간소화된 Flask 앱
- 데이터베이스 연결 제거
- API 키 의존성 제거
- 정적 페이지 중심으로 구성
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# 기본 설정
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'pixie-demo-key-2025')

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('dashboard.html')

@app.route('/survey')
def survey():
    """투자 성향 설문 페이지"""
    return render_template('survey.html')

@app.route('/chatbot')
def chatbot():
    """AI 챗봇 페이지"""
    return render_template('chatbot.html')

@app.route('/news')
def news():
    """뉴스/이슈 페이지"""
    return render_template('news.html')

@app.route('/stock')
def stock():
    """주식 분석 페이지"""
    return render_template('stock.html')

@app.route('/learning')
def learning():
    """투자 학습 페이지"""
    return render_template('learning.html')

@app.route('/alerts')
def alerts():
    """알림 페이지"""
    return render_template('alerts.html')

# API 엔드포인트 (데모용)
@app.route('/api/demo/chat', methods=['POST'])
def demo_chat():
    """데모 챗봇 응답"""
    return jsonify({
        'status': 'success',
        'response': '안녕하세요! Pixie AI 투자 어드바이저입니다. 현재 데모 모드로 실행 중입니다.',
        'agent': 'Pixie Demo'
    })

@app.route('/api/demo/survey', methods=['POST'])
def demo_survey():
    """데모 설문 결과"""
    return jsonify({
        'status': 'success',
        'profile': {
            'type': '안정형',
            'risk_score': 30,
            'description': '안정적인 투자를 선호하는 투자자입니다.'
        }
    })

@app.route('/health')
def health():
    """헬스체크 엔드포인트"""
    return jsonify({'status': 'healthy', 'service': 'Pixie Investment Advisor'})

if __name__ == '__main__':
    app.run(debug=False)