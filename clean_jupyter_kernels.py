"""중복된 Jupyter 커널 정리 스크립트"""
import os
import json
import shutil
import subprocess
import sys

def clean_jupyter_kernels():
    """중복된 Jupyter 커널을 정리하는 함수"""
    print("Jupyter 커널 정리 시작...")
    
    # 프로젝트 루트 디렉토리
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_path = os.path.join(project_root, "venv", "Scripts", "python.exe")
    
    # 1. 현재 설치된 커널 목록 확인
    print("\n1. 현재 설치된 커널 목록:")
    try:
        result = subprocess.run(["jupyter", "kernelspec", "list"], 
                              capture_output=True, text=True, shell=True)
        print(result.stdout)
    except:
        print("jupyter 명령을 찾을 수 없습니다. 계속 진행합니다.")
    
    # 2. 사용자별 커널 디렉토리 확인
    kernel_dirs = [
        os.path.expanduser("~/.jupyter/kernels"),
        os.path.expanduser("~/AppData/Roaming/jupyter/kernels"),
        os.path.expanduser("~/.local/share/jupyter/kernels")
    ]
    
    print("\n2. 커널 디렉토리 정리:")
    for kernel_dir in kernel_dirs:
        if os.path.exists(kernel_dir):
            print(f"\n디렉토리 확인: {kernel_dir}")
            for kernel_name in os.listdir(kernel_dir):
                kernel_path = os.path.join(kernel_dir, kernel_name)
                if os.path.isdir(kernel_path):
                    print(f"  - 발견된 커널: {kernel_name}")
                    
                    # kernel.json 확인
                    json_path = os.path.join(kernel_path, "kernel.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            kernel_spec = json.load(f)
                            print(f"    디스플레이 이름: {kernel_spec.get('display_name', 'N/A')}")
                            print(f"    Python 경로: {kernel_spec.get('argv', ['N/A'])[0]}")
    
    # 3. 모든 venv 관련 커널 제거
    print("\n3. 기존 venv 커널 모두 제거:")
    try:
        # jupyter kernelspec uninstall 명령 사용
        subprocess.run(["jupyter", "kernelspec", "uninstall", "-f", "venv"], 
                      shell=True, capture_output=True)
        print("기존 'venv' 커널 제거 완료")
    except:
        print("'venv' 커널이 없거나 제거 실패")
    
    # 수동으로도 제거
    for kernel_dir in kernel_dirs:
        if os.path.exists(kernel_dir):
            venv_path = os.path.join(kernel_dir, "venv")
            if os.path.exists(venv_path):
                shutil.rmtree(venv_path)
                print(f"디렉토리 제거: {venv_path}")
    
    # 4. VS Code 캐시 정리
    print("\n4. VS Code 캐시 정리:")
    vscode_dirs = [
        os.path.expanduser("~/.vscode-server"),
        os.path.expanduser("~/AppData/Roaming/Code/User/workspaceStorage"),
        os.path.expanduser("~/AppData/Roaming/Code/User/globalStorage/ms-toolsai.jupyter")
    ]
    
    for vscode_dir in vscode_dirs:
        if os.path.exists(vscode_dir):
            print(f"  - VS Code 캐시 디렉토리 발견: {vscode_dir}")
    
    # 5. 새로운 커널 설치 (하나만)
    print("\n5. 새로운 venv 커널 설치:")
    try:
        # 커널 재설치
        subprocess.run([python_path, "-m", "ipykernel", "install", 
                       "--user", "--name=venv", 
                       "--display-name=venv (Python 3.9.2)"], 
                      check=True)
        print("새 커널 설치 완료: venv (Python 3.9.2)")
    except Exception as e:
        print(f"커널 설치 실패: {e}")
        return False
    
    # 6. 설치 확인
    print("\n6. 설치 확인:")
    try:
        result = subprocess.run(["jupyter", "kernelspec", "list"], 
                              capture_output=True, text=True, shell=True)
        print(result.stdout)
    except:
        print("확인 실패")
    
    print("\n[완료] Jupyter 커널 정리 완료!")
    print("\n다음 단계:")
    print("1. VS Code를 완전히 종료하세요 (모든 창)")
    print("2. VS Code를 다시 시작하세요")
    print("3. 노트북을 열고 커널로 'venv (Python 3.9.2)'를 선택하세요")
    print("\n팁: 만약 여전히 중복이 보인다면:")
    print("  - VS Code 설정에서 'Jupyter: Kernel Spec Path'를 초기화하세요")
    print("  - Python 확장과 Jupyter 확장을 재설치하세요")
    
    return True

if __name__ == "__main__":
    success = clean_jupyter_kernels()
    if not success:
        print("\n[실패] 커널 정리 실패")
        sys.exit(1)