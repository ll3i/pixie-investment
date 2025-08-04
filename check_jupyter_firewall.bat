@echo off
echo Jupyter 포트 및 방화벽 확인 중...
echo.

echo 1. 사용 중인 포트 확인:
netstat -an | findstr :8888
netstat -an | findstr :9000
netstat -an | findstr :9005
echo.

echo 2. Windows 방화벽 규칙 추가 (관리자 권한 필요):
echo 관리자 권한으로 실행하려면 Y를 입력하세요.
set /p admin="계속하시겠습니까? (Y/N): "
if /i "%admin%"=="Y" (
    echo 방화벽 규칙 추가 중...
    netsh advfirewall firewall add rule name="Jupyter Notebook" dir=in action=allow protocol=TCP localport=8888
    netsh advfirewall firewall add rule name="IPython Kernel" dir=in action=allow protocol=TCP localport=8888-9999
    echo 방화벽 규칙 추가 완료!
) else (
    echo 방화벽 규칙 추가를 건너뜁니다.
)
echo.

echo 3. Python 프로세스 확인:
tasklist | findstr python
echo.

echo 완료! VS Code를 재시작해주세요.
pause