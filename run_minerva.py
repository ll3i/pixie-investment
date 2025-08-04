#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MINERVA 웹 애플리케이션 실행 스크립트
- 환경 설정 확인
- 시스템 초기화
- Flask 서버 시작
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def check_environment():
    """환경 설정 확인"""
    required_vars = ['FLASK_SECRET_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ 다음 환경 변수가 설정되지 않았습니다:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n.env 파일을 생성하고 필요한 환경 변수를 설정해주세요.")
        print("예시:")
        print("FLASK_SECRET_KEY=your_32_character_or_longer_secret_key")
        print("OPENAI_API_KEY=your_openai_api_key")
        print("CLOVA_API_KEY=your_clova_api_key")
        return False
    
    # API 키 확인
    api_keys = {
        'OpenAI': os.environ.get('OPENAI_API_KEY'),
        'CLOVA': os.environ.get('CLOVA_API_KEY')
    }
    
    available_apis = [name for name, key in api_keys.items() if key]
    
    if not available_apis:
        print("⚠️  API 키가 설정되지 않았습니다. 시뮬레이션 모드로 실행됩니다.")
    else:
        print(f"✅ 사용 가능한 API: {', '.join(available_apis)}")
    
    return True

def main():
    """메인 실행 함수"""
    print("🚀 MINERVA 투자 챗봇 시스템 시작")
    print("=" * 50)
    
    # 환경 설정 확인
    if not check_environment():
        sys.exit(1)
    
    # Flask 애플리케이션 실행
    try:
        from app_v2 import app
        
        print("\n✅ 환경 설정 완료")
        print("📊 웹 서버 시작 중...")
        print("🌐 브라우저에서 http://localhost:5000 으로 접속하세요")
        print("\n종료하려면 Ctrl+C를 누르세요")
        print("=" * 50)
        
        # Flask 서버 실행
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=os.environ.get('DEBUG', 'false').lower() == 'true'
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 서버가 중지되었습니다.")
    except Exception as e:
        print(f"\n❌ 서버 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 