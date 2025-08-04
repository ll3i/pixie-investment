"""
Waitress를 사용한 안정적인 프로덕션 서버
"""
import os
import sys
from app import app

try:
    from waitress import serve
except ImportError:
    print("Waitress가 설치되어 있지 않습니다.")
    print("다음 명령어로 설치해주세요:")
    print("pip install waitress")
    sys.exit(1)

if __name__ == '__main__':
    print("Pixie 투자챗봇 웹서비스를 시작합니다...")
    print("브라우저에서 http://localhost:8080 으로 접속하세요")
    print("종료하려면 Ctrl+C를 누르세요")
    print("\n[Waitress 프로덕션 서버 사용 중]")
    
    # Waitress 서버 실행 (Windows에서 더 안정적)
    serve(
        app,
        host='127.0.0.1',
        port=8080,
        threads=4,
        connection_limit=100,
        cleanup_interval=30,
        channel_timeout=120
    )