"""
간단하고 안정적인 서버 시작 스크립트
"""
import os
import sys
import socket
import time

# Python path 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from app import app

def kill_process_on_port(port):
    """포트를 사용 중인 프로세스 종료"""
    try:
        import subprocess
        # Windows netstat 명령으로 포트 사용 프로세스 찾기
        result = subprocess.run(
            f'netstat -ano | findstr :{port}', 
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'LISTENING' in line:
                    parts = line.split()
                    pid = parts[-1]
                    print(f"포트 {port}를 사용 중인 프로세스(PID: {pid}) 종료 시도...")
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True)
                    time.sleep(1)
    except Exception as e:
        print(f"프로세스 종료 중 오류: {e}")

def is_port_available(port):
    """포트가 사용 가능한지 확인"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return True
    except:
        return False

def start_server():
    """서버 시작"""
    ports = [8080, 8000, 5000, 8888, 9000]
    
    print("Pixie 투자챗봇 웹서비스를 시작합니다...")
    
    # 사용 가능한 포트 찾기
    selected_port = None
    for port in ports:
        if is_port_available(port):
            selected_port = port
            break
        else:
            print(f"포트 {port}가 사용 중입니다.")
            # 포트를 사용 중인 프로세스 종료 시도
            response = input(f"포트 {port}를 사용 중인 프로세스를 종료하시겠습니까? (y/n): ")
            if response.lower() == 'y':
                kill_process_on_port(port)
                if is_port_available(port):
                    selected_port = port
                    break
    
    if not selected_port:
        print("사용 가능한 포트를 찾을 수 없습니다.")
        return
    
    print(f"\n포트 {selected_port}에서 서버를 시작합니다...")
    print(f"브라우저에서 http://localhost:{selected_port} 으로 접속하세요")
    print("종료하려면 Ctrl+C를 누르세요\n")
    
    # Flask 앱 설정
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    try:
        # 가장 기본적인 Flask 서버 실행
        app.run(
            host='127.0.0.1',
            port=selected_port,
            debug=False,
            use_reloader=False,
            threaded=False  # 단일 스레드로 실행
        )
    except Exception as e:
        print(f"\n서버 실행 중 오류 발생: {e}")
        print("\n다른 방법을 시도해보세요:")
        print("1. 컴퓨터를 재시작하고 다시 시도")
        print("2. 관리자 권한으로 실행")
        print("3. Windows Defender/안티바이러스 일시 중지")

if __name__ == '__main__':
    start_server()