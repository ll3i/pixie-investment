"""
안정적인 웹서버 시작 스크립트
"""
import os
import sys
import time
from werkzeug.serving import run_simple
from app import app

def start_server():
    """안정적인 서버 시작 함수"""
    print("Pixie 투자챗봇 웹서비스를 시작합니다...")
    print("브라우저에서 http://localhost:8080 으로 접속하세요")
    print("종료하려면 Ctrl+C를 누르세요")
    
    # 포트 충돌 방지를 위한 대기
    time.sleep(1)
    
    try:
        # Werkzeug의 run_simple 사용 (더 안정적)
        run_simple(
            'localhost',
            8080,
            app,
            use_reloader=False,
            use_debugger=False,
            threaded=True  # 멀티스레드 활성화
        )
    except OSError as e:
        if "Address already in use" in str(e) or "이미 사용" in str(e):
            print("\n포트 8080이 이미 사용 중입니다.")
            print("다른 포트(8000)로 재시도합니다...")
            time.sleep(2)
            run_simple(
                'localhost',
                8000,
                app,
                use_reloader=False,
                use_debugger=False,
                threaded=True
            )
        else:
            raise e
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\n대체 방법으로 시작합니다...")
        # Flask 기본 서버로 fallback
        app.run(
            host='127.0.0.1',  # localhost 대신 명시적 IP 사용
            port=8080,
            debug=False,
            use_reloader=False,
            threaded=True
        )

if __name__ == '__main__':
    start_server()