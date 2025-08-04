"""
포트 사용 상태 확인 스크립트
"""
import socket
import psutil

def check_port(port):
    """특정 포트가 사용 중인지 확인"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

def find_process_using_port(port):
    """특정 포트를 사용하는 프로세스 찾기"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                process = psutil.Process(conn.pid)
                return f"PID: {conn.pid}, 프로세스: {process.name()}"
            except:
                return f"PID: {conn.pid}"
    return None

if __name__ == "__main__":
    ports_to_check = [8080, 8000, 5000]
    
    print("포트 사용 상태 확인\n")
    for port in ports_to_check:
        if check_port(port):
            process_info = find_process_using_port(port)
            print(f"포트 {port}: 사용 중")
            if process_info:
                print(f"  └─ {process_info}")
        else:
            print(f"포트 {port}: 사용 가능")
    
    print("\n팁: 포트가 사용 중이면 해당 프로세스를 종료하거나 다른 포트를 사용하세요.")