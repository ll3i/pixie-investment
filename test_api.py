#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""API 엔드포인트 테스트"""

import requests
import json

def test_evaluation_insights():
    """평가 인사이트 API 테스트"""
    try:
        # 로컬 서버 URL
        url = "http://localhost:5000/api/evaluation-insights"
        
        # GET 요청
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Data length: {len(data.get('data', []))}")
            
            # 첫 번째 인사이트 출력
            if data.get('data'):
                first_insight = data['data'][0]
                print("\n첫 번째 인사이트:")
                for key, value in first_insight.items():
                    print(f"  {key}: {value}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("서버에 연결할 수 없습니다. Flask 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    print("평가 인사이트 API 테스트 시작...")
    test_evaluation_insights()