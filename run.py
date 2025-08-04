#!/usr/bin/env python3
"""
투자챗봇 서비스 실행 스크립트
"""

import os
import sys
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# Python path 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 기본 환경변수 설정
if not os.environ.get("FLASK_SECRET_KEY"):
    os.environ["FLASK_SECRET_KEY"] = "minerva_investment_advisor_secure_key_2024"
if not os.environ.get("FLASK_ENV"):
    os.environ["FLASK_ENV"] = "development"

from app import app

if __name__ == '__main__':
    print("🚀 투자챗봇 서비스를 시작합니다...")
    print("📱 웹 브라우저에서 http://localhost:5000 으로 접속하세요")
    print("⚠️  OpenAI API 키가 설정되지 않은 경우 일부 기능이 제한될 수 있습니다.")
    
    # 개발 모드로 실행
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    ) 