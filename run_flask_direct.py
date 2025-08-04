"""
직접 Flask 개발 서버 실행 (Windows 호환성을 위한 대안)
"""
from app import app
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Pixie 투자챗봇 웹서비스를 시작합니다... (Flask 개발 서버)")
    print(f"주소: http://localhost:{port}")
    print("Ctrl+C를 눌러 종료하세요.")
    
    try:
        # Flask 개발 서버 직접 사용
        app.run(host='127.0.0.1', port=port, debug=False)
    except KeyboardInterrupt:
        print("\n서버를 종료합니다...")
    except Exception as e:
        print(f"서버 실행 중 오류 발생: {e}")