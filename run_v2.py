#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MINERVA 투자 분석 웹 애플리케이션 실행 스크립트 (V2)
- 서비스 알고리즘 흐름도에 맞게 재구성된 버전
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 현재 디렉토리를 파이썬 경로에 추가하여 src 모듈을 import할 수 있게 함
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'src'))

def run_app():
    """웹 애플리케이션을 실행합니다."""
    try:
        # 기본 포트 설정
        port = int(os.environ.get('PORT', 5000))
        
        # 디버그 모드 설정 - 디버깅을 위해 강제로 True로 설정
        debug = True
        print(f"디버그 모드가 활성화되었습니다.")
        
        # app_v2 모듈 임포트
        from app_v2 import app
        
        print(f"MINERVA 투자 분석 웹 애플리케이션 (V2)을 실행합니다. (포트: {port}, 디버그 모드: {debug})")
        
        # 애플리케이션 실행
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        print(f"애플리케이션 실행 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_app() 