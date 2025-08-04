"""Jupyter 완전 재설치 및 초기화 스크립트"""
import os
import shutil
import subprocess
import sys
import time

def clean_jupyter_completely():
    """Jupyter 관련 모든 설정과 캐시를 삭제하고 재설치"""
    print("Jupyter 완전 초기화 시작...")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_path = os.path.join(project_root, "venv", "Scripts", "python.exe")
    
    # 1. Jupyter 관련 디렉토리 모두 삭제
    dirs_to_clean = [
        os.path.expanduser("~/.jupyter"),
        os.path.expanduser("~/.ipython"),
        os.path.expanduser("~/AppData/Roaming/jupyter"),
        os.path.expanduser("~/.local/share/jupyter"),
        os.path.expanduser("~/AppData/Local/jupyter"),
    ]
    
    print("\n1. Jupyter 설정 디렉토리 정리:")
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"   삭제됨: {dir_path}")
            except Exception as e:
                print(f"   삭제 실패: {dir_path} - {e}")
    
    # 2. 커널 제거
    print("\n2. 기존 커널 제거:")
    try:
        subprocess.run(["jupyter", "kernelspec", "list"], shell=True, capture_output=True)
        subprocess.run(["jupyter", "kernelspec", "remove", "-f", "venv"], shell=True, capture_output=True)
        print("   커널 제거 완료")
    except:
        print("   커널이 없거나 제거 실패")
    
    # 3. Jupyter 관련 패키지 재설치
    print("\n3. Jupyter 패키지 재설치:")
    packages = ["ipykernel", "jupyter", "notebook", "jupyter_client", "jupyter_core"]
    
    # 제거
    print("   패키지 제거 중...")
    for pkg in packages:
        subprocess.run([python_path, "-m", "pip", "uninstall", "-y", pkg], 
                      capture_output=True)
    
    # 재설치
    print("   패키지 설치 중...")
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade"] + packages, 
                  check=True)
    print("   패키지 설치 완료")
    
    # 4. 새로운 설정 파일 생성 (IPv4 전용)
    print("\n4. 새 설정 파일 생성:")
    
    # Jupyter 디렉토리 생성
    jupyter_dir = os.path.expanduser("~/.jupyter")
    os.makedirs(jupyter_dir, exist_ok=True)
    
    # jupyter_notebook_config.py
    config_path = os.path.join(jupyter_dir, "jupyter_notebook_config.py")
    config_content = """# Jupyter Configuration for Windows
c = get_config()

# Force IPv4
c.NotebookApp.ip = '127.0.0.1'
c.NotebookApp.port = 8888
c.NotebookApp.port_retries = 50
c.NotebookApp.open_browser = False

# Kernel settings
c.KernelManager.ip = '127.0.0.1'
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"   설정 파일 생성: {config_path}")
    
    # 5. 커널 재설치 (간단한 설정으로)
    print("\n5. 커널 재설치:")
    try:
        # 환경 변수 설정
        env = os.environ.copy()
        env['JUPYTER_PREFER_ENV_PATH'] = '0'
        env['JUPYTER_IP'] = '127.0.0.1'
        
        subprocess.run([python_path, "-m", "ipykernel", "install", 
                       "--user", "--name=venv", 
                       "--display-name=venv (Python 3.9.2)"], 
                      env=env, check=True)
        print("   커널 설치 완료")
    except Exception as e:
        print(f"   커널 설치 실패: {e}")
        return False
    
    # 6. VS Code 설정 간소화
    vscode_settings_dir = os.path.join(project_root, ".vscode")
    os.makedirs(vscode_settings_dir, exist_ok=True)
    
    settings_path = os.path.join(vscode_settings_dir, "settings.json")
    settings = {
        "python.defaultInterpreterPath": "${workspaceFolder}\\venv\\Scripts\\python.exe",
        "jupyter.jupyterServerType": "local"
    }
    
    import json
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    print(f"\n6. VS Code 설정 파일 생성: {settings_path}")
    
    print("\n[완료] Jupyter 완전 재설치 완료!")
    print("\n다음 단계:")
    print("1. VS Code를 완전히 종료")
    print("2. 작업 관리자에서 Code.exe 프로세스 확인 및 종료")
    print("3. VS Code 재시작")
    print("4. 노트북 파일 열기")
    print("5. 커널 선택: 'venv (Python 3.9.2)'")
    print("\n팁: 만약 여전히 문제가 있다면 Python을 3.9.2에서 최신 버전으로 업그레이드를 고려하세요.")
    
    return True

if __name__ == "__main__":
    # 관리자 권한 확인
    import ctypes
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("주의: 관리자 권한으로 실행하면 더 완전한 정리가 가능합니다.")
    
    success = clean_jupyter_completely()
    if not success:
        print("\n[실패] 재설치 실패")
        sys.exit(1)