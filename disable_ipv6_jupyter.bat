@echo off
echo Jupyter용 IPv6 비활성화 및 환경 설정...
echo.

REM 환경 변수 설정
set JUPYTER_IP=127.0.0.1
set IPYTHON_KERNEL_IP=127.0.0.1
set JUPYTER_DISABLE_IPV6=1
set JUPYTER_PREFER_ENV_PATH=1
set IPY_INTERRUPT_EVENT=1
set PYDEVD_DISABLE_FILE_VALIDATION=1

echo 환경 변수 설정 완료
echo JUPYTER_IP=%JUPYTER_IP%
echo IPYTHON_KERNEL_IP=%IPYTHON_KERNEL_IP%
echo JUPYTER_DISABLE_IPV6=%JUPYTER_DISABLE_IPV6%
echo.

REM VS Code 실행 (환경 변수 적용된 상태로)
echo VS Code를 IPv4 전용 모드로 실행합니다...
start "" "C:\Program Files\Microsoft VS Code\Code.exe" "%~dp0.."

echo.
echo VS Code가 실행되었습니다.
echo Jupyter 노트북을 열고 커널을 실행해보세요.
pause