"""
Waitress 서버를 사용한 Flask 앱 실행
Windows에서 안정적인 실행을 위한 대안
"""
from waitress import serve
from app import app
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')  # localhost only for Windows compatibility
    
    print(f"Pixie 투자챗봇 웹서비스를 시작합니다... (Waitress 서버)")
    print(f"주소: http://localhost:{port}")
    print("Ctrl+C를 눌러 종료하세요.")
    
    try:
        # Windows에서 더 안전한 설정 사용
        serve(app, host=host, port=port, threads=4, 
              connection_limit=100,
              cleanup_interval=30,
              channel_timeout=60)
    except KeyboardInterrupt:
        print("\n서버를 종료합니다...")
    except Exception as e:
        print(f"서버 실행 중 오류 발생: {e}")