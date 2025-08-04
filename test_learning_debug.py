"""
학습 페이지 디버깅을 위한 독립 실행 테스트 스크립트
"""
import os
import sys

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def test_template_files():
    """템플릿 파일 존재 여부 확인"""
    print("=== 템플릿 파일 확인 ===")
    template_dir = os.path.join(current_dir, 'templates')
    print(f"템플릿 디렉토리: {template_dir}")
    print(f"디렉토리 존재: {os.path.exists(template_dir)}")
    
    if os.path.exists(template_dir):
        templates = ['learning.html', 'layout.html', 'dashboard.html']
        for template in templates:
            path = os.path.join(template_dir, template)
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            readable = os.access(path, os.R_OK) if exists else False
            print(f"\n{template}:")
            print(f"  - 경로: {path}")
            print(f"  - 존재: {exists}")
            print(f"  - 크기: {size} bytes")
            print(f"  - 읽기 가능: {readable}")

def test_flask_app():
    """Flask 앱 초기화 테스트"""
    print("\n=== Flask 앱 테스트 ===")
    try:
        from app import app
        print(f"Flask 앱 로드 성공")
        print(f"템플릿 폴더: {app.template_folder}")
        print(f"정적 파일 폴더: {app.static_folder}")
        
        # 라우트 확인
        print("\n학습 관련 라우트:")
        for rule in app.url_map.iter_rules():
            if 'learning' in rule.rule:
                print(f"  - {rule.rule} [{', '.join(rule.methods - {'HEAD', 'OPTIONS'})}]")
                
    except Exception as e:
        print(f"Flask 앱 로드 실패: {e}")
        import traceback
        traceback.print_exc()

def test_render_template():
    """템플릿 렌더링 테스트"""
    print("\n=== 템플릿 렌더링 테스트 ===")
    try:
        from flask import Flask, render_template
        from app import app
        
        with app.app_context():
            # 간단한 템플릿 렌더링 시도
            try:
                # learning.html 렌더링 시도
                html = render_template('learning.html')
                print("✅ learning.html 렌더링 성공!")
                print(f"   렌더링된 HTML 크기: {len(html)} bytes")
            except Exception as e:
                print(f"❌ learning.html 렌더링 실패: {e}")
                
            # layout.html 렌더링 시도  
            try:
                html = render_template('layout.html')
                print("✅ layout.html 렌더링 성공!")
            except Exception as e:
                print(f"❌ layout.html 렌더링 실패: {e}")
                
    except Exception as e:
        print(f"렌더링 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def test_static_files():
    """정적 파일 확인"""
    print("\n=== 정적 파일 확인 ===")
    static_dir = os.path.join(current_dir, 'static')
    print(f"정적 파일 디렉토리: {static_dir}")
    print(f"디렉토리 존재: {os.path.exists(static_dir)}")
    
    if os.path.exists(static_dir):
        # 이미지 파일 확인
        img_dir = os.path.join(static_dir, 'images')
        if os.path.exists(img_dir):
            print(f"\n이미지 디렉토리: {img_dir}")
            for file in os.listdir(img_dir):
                if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    path = os.path.join(img_dir, file)
                    size = os.path.getsize(path)
                    print(f"  - {file} ({size} bytes)")

def main():
    """메인 테스트 실행"""
    print("학습 페이지 디버깅 시작...\n")
    
    # 1. 템플릿 파일 확인
    test_template_files()
    
    # 2. Flask 앱 테스트
    test_flask_app()
    
    # 3. 템플릿 렌더링 테스트
    test_render_template()
    
    # 4. 정적 파일 확인
    test_static_files()
    
    print("\n=== 테스트 완료 ===")
    print("\n디버깅 팁:")
    print("1. /test/learning-minimal 접속 - 가장 기본적인 라우팅 테스트")
    print("2. /test/template-check 접속 - 템플릿 파일 존재 여부 JSON 확인")
    print("3. /test/learning 접속 - 상세 디버그 정보 확인")
    print("4. /test/learning-with-layout 접속 - 레이아웃 상속 테스트")

if __name__ == "__main__":
    main()