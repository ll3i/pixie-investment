"""Jupyter 커널 연결 문제 해결 스크립트"""
import subprocess
import sys
import os

def fix_jupyter_kernel():
    """Jupyter 커널 문제를 해결하는 함수"""
    print("Jupyter 커널 문제 해결 시작...")
    
    # 프로젝트 루트 디렉토리로 이동
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # venv Python 경로
    python_path = os.path.join(project_root, "venv", "Scripts", "python.exe")
    
    if not os.path.exists(python_path):
        print(f"오류: Python 경로를 찾을 수 없습니다: {python_path}")
        return False
    
    print(f"Python 경로: {python_path}")
    
    # 1. 기존 ipykernel 제거
    print("\n1. 기존 ipykernel 제거 중...")
    try:
        subprocess.run([python_path, "-m", "pip", "uninstall", "-y", "ipykernel"], check=True)
        print("ipykernel 제거 완료")
    except:
        print("ipykernel이 설치되어 있지 않습니다.")
    
    # 2. ipykernel 재설치
    print("\n2. ipykernel 재설치 중...")
    try:
        subprocess.run([python_path, "-m", "pip", "install", "ipykernel"], check=True)
        print("ipykernel 설치 완료")
    except Exception as e:
        print(f"ipykernel 설치 실패: {e}")
        return False
    
    # 3. Jupyter 커널 재설정
    print("\n3. Jupyter 커널 재설정 중...")
    try:
        subprocess.run([python_path, "-m", "ipykernel", "install", "--user", "--name=venv", "--display-name=venv"], check=True)
        print("Jupyter 커널 설정 완료")
    except Exception as e:
        print(f"커널 설정 실패: {e}")
        return False
    
    # 4. 포트 충돌 확인 및 해결
    print("\n4. 포트 충돌 확인 중...")
    try:
        # 9005 포트 사용 프로세스 확인
        result = subprocess.run(["netstat", "-ano", "|", "findstr", ":9005"], 
                              shell=True, capture_output=True, text=True)
        if result.stdout:
            print("포트 9005가 사용 중입니다.")
            print("VS Code를 재시작하여 다른 포트를 사용하도록 하세요.")
        else:
            print("포트 9005가 사용 가능합니다.")
    except:
        print("포트 확인 실패")
    
    # 5. Jupyter 설정 파일 생성
    print("\n5. Jupyter 설정 파일 생성 중...")
    jupyter_config_dir = os.path.expanduser("~/.jupyter")
    os.makedirs(jupyter_config_dir, exist_ok=True)
    
    config_path = os.path.join(jupyter_config_dir, "jupyter_notebook_config.py")
    config_content = """# Jupyter 설정
c.NotebookApp.ip = 'localhost'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
"""
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"Jupyter 설정 파일 생성 완료: {config_path}")
    
    print("\n[성공] Jupyter 커널 문제 해결 완료!")
    print("\n다음 단계:")
    print("1. VS Code를 완전히 종료하고 다시 시작하세요.")
    print("2. 노트북 파일을 다시 열어보세요.")
    print("3. 커널 선택 시 'venv'를 선택하세요.")
    
    return True

if __name__ == "__main__":
    success = fix_jupyter_kernel()
    if not success:
        print("\n[실패] 문제 해결 실패. 수동으로 해결이 필요합니다.")
        sys.exit(1)