"""Jupyter IPv6 문제 해결 스크립트 - IPv4 전용 설정"""
import os
import json
import subprocess
import sys

def fix_jupyter_ipv6_issue():
    """Jupyter의 IPv6 바인딩 문제를 해결하고 IPv4만 사용하도록 설정"""
    print("Jupyter IPv6 문제 해결 시작...")
    
    # 1. 환경 변수 설정 파일 생성
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_script = os.path.join(project_root, "set_jupyter_env.bat")
    
    env_content = """@echo off
REM Jupyter를 IPv4 전용으로 설정
set JUPYTER_PREFER_ENV_PATH=1
set JUPYTER_IP=127.0.0.1
set IPYTHON_KERNEL_IP=127.0.0.1
set JUPYTER_DISABLE_IPV6=1
set IPY_INTERRUPT_EVENT=1
set PYDEVD_DISABLE_FILE_VALIDATION=1

echo Jupyter 환경 변수 설정 완료
"""
    
    with open(env_script, 'w', encoding='utf-8') as f:
        f.write(env_content)
    print(f"환경 변수 스크립트 생성: {env_script}")
    
    # 2. Jupyter 설정 파일 업데이트 (IPv4 전용)
    jupyter_dir = os.path.expanduser("~/.jupyter")
    os.makedirs(jupyter_dir, exist_ok=True)
    
    # jupyter_notebook_config.py
    config_path = os.path.join(jupyter_dir, "jupyter_notebook_config.py")
    config_content = """# Jupyter Notebook 설정 - IPv4 전용
c = get_config()

# IPv4 전용 설정
c.NotebookApp.ip = '127.0.0.1'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_remote_access = False

# 커널 매니저 설정
c.KernelManager.ip = '127.0.0.1'
c.KernelManager.transport = 'tcp'
c.KernelManager.kernel_cmd = [
    'python', '-m', 'ipykernel_launcher', 
    '-f', '{connection_file}',
    '--IPKernelApp.ip=127.0.0.1'
]

# Connection 파일 설정
c.ConnectionFileMixin.ip = '127.0.0.1'
c.ConnectionFileMixin.transport = 'tcp'
"""
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"Jupyter 설정 파일 업데이트: {config_path}")
    
    # 3. IPython 커널 설정 (IPv4 전용)
    ipython_dir = os.path.expanduser("~/.ipython/profile_default")
    os.makedirs(ipython_dir, exist_ok=True)
    
    kernel_config_path = os.path.join(ipython_dir, "ipython_kernel_config.py")
    kernel_config_content = """# IPython 커널 설정 - IPv4 전용
c = get_config()

# IPv4 전용 커널 설정
c.IPKernelApp.ip = '127.0.0.1'
c.IPKernelApp.transport = 'tcp'
c.IPKernelApp.hb_port = 0
c.IPKernelApp.shell_port = 0
c.IPKernelApp.iopub_port = 0
c.IPKernelApp.stdin_port = 0
c.IPKernelApp.control_port = 0

# 연결 파일 설정
c.IPKernelApp.connection_file = ''
"""
    
    with open(kernel_config_path, 'w', encoding='utf-8') as f:
        f.write(kernel_config_content)
    print(f"IPython 커널 설정 파일 업데이트: {kernel_config_path}")
    
    # 4. 커널 스펙 파일 직접 수정
    kernel_dir = os.path.expanduser("~/AppData/Roaming/jupyter/kernels/venv")
    if os.path.exists(kernel_dir):
        kernel_json_path = os.path.join(kernel_dir, "kernel.json")
        if os.path.exists(kernel_json_path):
            with open(kernel_json_path, 'r', encoding='utf-8') as f:
                kernel_spec = json.load(f)
            
            # argv에 IP 설정 추가
            python_path = kernel_spec["argv"][0]
            kernel_spec["argv"] = [
                python_path,
                "-m", "ipykernel_launcher",
                "-f", "{connection_file}",
                "--IPKernelApp.ip=127.0.0.1"
            ]
            kernel_spec["env"] = {
                "JUPYTER_IP": "127.0.0.1",
                "IPYTHON_KERNEL_IP": "127.0.0.1",
                "JUPYTER_DISABLE_IPV6": "1"
            }
            
            with open(kernel_json_path, 'w', encoding='utf-8') as f:
                json.dump(kernel_spec, f, indent=2)
            print(f"커널 스펙 파일 업데이트: {kernel_json_path}")
    
    # 5. VS Code 설정 업데이트
    vscode_settings_dir = os.path.join(project_root, ".vscode")
    os.makedirs(vscode_settings_dir, exist_ok=True)
    
    settings_path = os.path.join(vscode_settings_dir, "settings.json")
    settings = {
        "jupyter.jupyterServerType": "local",
        "jupyter.notebookFileRoot": "${workspaceFolder}",
        "python.defaultInterpreterPath": "${workspaceFolder}\\venv\\Scripts\\python.exe",
        "jupyter.askForKernelRestart": False,
        "jupyter.jupyterLaunchTimeout": 60000,
        "jupyter.jupyterLaunchRetries": 3,
        "terminal.integrated.env.windows": {
            "JUPYTER_IP": "127.0.0.1",
            "IPYTHON_KERNEL_IP": "127.0.0.1",
            "JUPYTER_DISABLE_IPV6": "1"
        }
    }
    
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    print(f"VS Code 설정 파일 업데이트: {settings_path}")
    
    # 6. 시스템 환경 변수 설정 (현재 세션)
    os.environ['JUPYTER_IP'] = '127.0.0.1'
    os.environ['IPYTHON_KERNEL_IP'] = '127.0.0.1'
    os.environ['JUPYTER_DISABLE_IPV6'] = '1'
    
    print("\n[완료] Jupyter IPv6 문제 해결!")
    print("\n중요한 다음 단계:")
    print("1. 명령 프롬프트를 관리자 권한으로 실행")
    print("2. 다음 명령 실행:")
    print(f"   cd {project_root}")
    print("   set_jupyter_env.bat")
    print("3. VS Code를 완전히 종료 (모든 창)")
    print("4. VS Code를 환경 변수가 설정된 상태에서 재시작")
    print("5. 노트북 파일을 열고 커널 실행")
    
    # 7. 간단한 테스트 스크립트 생성
    test_script = os.path.join(project_root, "test_jupyter_connection.py")
    test_content = """import socket

# IPv4 연결 테스트
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    print(f"[성공] IPv4 연결 가능 - 포트 {port}")
except Exception as e:
    print(f"[실패] IPv4 연결 오류: {e}")

# IPv6 연결 테스트 (실패해야 정상)
try:
    s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s6.bind(('::1', 0))
    port = s6.getsockname()[1]
    s6.close()
    print(f"[경고] IPv6 연결 가능 - 포트 {port}")
except Exception as e:
    print(f"[정상] IPv6 연결 차단됨: {e}")
"""
    
    with open(test_script, 'w', encoding='utf-8') as f:
        f.write(test_content)
    print(f"\n테스트 스크립트 생성: {test_script}")
    print("테스트 실행: python test_jupyter_connection.py")
    
    return True

if __name__ == "__main__":
    success = fix_jupyter_ipv6_issue()
    if not success:
        print("\n[실패] 문제 해결 실패")
        sys.exit(1)