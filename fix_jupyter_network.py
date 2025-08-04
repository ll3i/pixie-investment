"""Jupyter 네트워크 바인딩 문제 해결 스크립트"""
import os
import json
import subprocess
import sys

def fix_jupyter_network_binding():
    """Jupyter의 0.0.0.0 바인딩 문제를 해결"""
    print("Jupyter 네트워크 바인딩 문제 해결 시작...")
    
    # 1. Jupyter 설정 디렉토리 생성
    jupyter_dir = os.path.expanduser("~/.jupyter")
    os.makedirs(jupyter_dir, exist_ok=True)
    
    # 2. jupyter_notebook_config.py 생성/수정
    config_path = os.path.join(jupyter_dir, "jupyter_notebook_config.py")
    config_content = """# Jupyter Notebook 설정
c = get_config()

# localhost 사용 강제 (0.0.0.0 대신)
c.NotebookApp.ip = '127.0.0.1'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False

# 커널 매니저 설정
c.KernelManager.ip = '127.0.0.1'

# Connection 파일 설정
c.ConnectionFileMixin.ip = '127.0.0.1'
"""
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"Jupyter 설정 파일 생성: {config_path}")
    
    # 3. jupyter_server_config.py 생성 (새 버전용)
    server_config_path = os.path.join(jupyter_dir, "jupyter_server_config.py")
    server_config_content = """# Jupyter Server 설정
c = get_config()

# localhost 사용 강제
c.ServerApp.ip = '127.0.0.1'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
"""
    
    with open(server_config_path, 'w', encoding='utf-8') as f:
        f.write(server_config_content)
    print(f"Jupyter Server 설정 파일 생성: {server_config_path}")
    
    # 4. ipython 프로파일 설정
    ipython_dir = os.path.expanduser("~/.ipython/profile_default")
    os.makedirs(ipython_dir, exist_ok=True)
    
    ipython_config_path = os.path.join(ipython_dir, "ipython_kernel_config.py")
    ipython_config_content = """# IPython 커널 설정
c = get_config()

# 커널 연결 설정
c.IPKernelApp.ip = '127.0.0.1'
c.IPKernelApp.transport = 'tcp'
"""
    
    with open(ipython_config_path, 'w', encoding='utf-8') as f:
        f.write(ipython_config_content)
    print(f"IPython 커널 설정 파일 생성: {ipython_config_path}")
    
    # 5. VS Code 설정 업데이트
    vscode_settings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".vscode")
    os.makedirs(vscode_settings_dir, exist_ok=True)
    
    settings_path = os.path.join(vscode_settings_dir, "settings.json")
    settings = {
        "jupyter.jupyterServerType": "local",
        "jupyter.notebookFileRoot": "${workspaceFolder}",
        "python.defaultInterpreterPath": "${workspaceFolder}\\venv\\Scripts\\python.exe",
        "jupyter.askForKernelRestart": False,
        "jupyter.widgetScriptSources": ["jsdelivr.com", "unpkg.com"],
        "jupyter.kernels.filter": [
            {
                "path": "${workspaceFolder}\\venv\\Scripts\\python.exe",
                "type": "pythonEnvironment"
            }
        ],
        "jupyter.jupyterLaunchTimeout": 60000,
        "jupyter.jupyterLaunchRetries": 3
    }
    
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    print(f"VS Code 설정 파일 업데이트: {settings_path}")
    
    # 6. 환경 변수 설정
    print("\n환경 변수 설정 중...")
    os.environ['JUPYTER_PREFER_ENV_PATH'] = '1'
    os.environ['JUPYTER_IP'] = '127.0.0.1'
    
    # 7. 커널 재설치
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_path = os.path.join(project_root, "venv", "Scripts", "python.exe")
    
    print("\n커널 재설치 중...")
    try:
        # 기존 커널 제거
        subprocess.run(["jupyter", "kernelspec", "remove", "-f", "venv"], 
                      shell=True, capture_output=True)
        
        # 새 커널 설치
        subprocess.run([python_path, "-m", "ipykernel", "install", 
                       "--user", "--name=venv", 
                       "--display-name=venv (Python 3.9.2)"], 
                      check=True)
        print("커널 재설치 완료")
    except Exception as e:
        print(f"커널 재설치 중 오류: {e}")
    
    print("\n[완료] Jupyter 네트워크 바인딩 문제 해결!")
    print("\n중요: 다음 단계를 반드시 수행하세요:")
    print("1. VS Code를 완전히 종료 (모든 창)")
    print("2. Windows 작업 관리자에서 Code.exe 프로세스가 없는지 확인")
    print("3. VS Code 재시작")
    print("4. 노트북 파일 열기")
    print("5. 커널 선택 시 'venv (Python 3.9.2)' 선택")
    
    return True

if __name__ == "__main__":
    success = fix_jupyter_network_binding()
    if not success:
        print("\n[실패] 문제 해결 실패")
        sys.exit(1)