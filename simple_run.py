"""
간단한 Flask 실행 스크립트
최소한의 설정으로 실행
"""
import os
import sys

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 환경 변수 설정
os.environ['WERKZEUG_RUN_MAIN'] = 'true'

from app import app

if __name__ == '__main__':
    print("Pixie 투자챗봇 웹서비스를 시작합니다...")
    print("주소: http://localhost:5000")
    print("Ctrl+C를 눌러 종료하세요.")
    
    try:
        # 가장 기본적인 설정으로 실행
        from werkzeug.serving import run_simple
        run_simple('localhost', 5000, app, use_reloader=False, use_debugger=False)
    except Exception as e:
        print(f"werkzeug 실행 실패: {e}")
        # 대체 방법
        app.run(host='localhost', port=5000, debug=False, use_reloader=False)