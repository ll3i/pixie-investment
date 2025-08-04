"""
간단한 웹서버 시작 스크립트
"""
from app import app

if __name__ == '__main__':
    print("Pixie 투자챗봇 웹서비스를 시작합니다...")
    print("브라우저에서 http://localhost:8080 으로 접속하세요")
    print("종료하려면 Ctrl+C를 누르세요")
    
    # Flask 개발 서버로 실행 (다른 포트 시도)
    try:
        app.run(host='localhost', port=8080, debug=False, use_reloader=False)
    except Exception as e:
        print(f"오류 발생: {e}")
        # 대체 포트로 시도
        print("포트 8000으로 다시 시도합니다...")
        app.run(host='localhost', port=8000, debug=False, use_reloader=False)