@echo off
echo Pixie 투자챗봇 웹서비스 시작...
echo.

REM Python 경로 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo Python이 설치되어 있지 않거나 PATH에 없습니다.
    pause
    exit /b 1
)

REM 환경 변수 설정
set FLASK_APP=app.py
set FLASK_ENV=production
set PYTHONIOENCODING=utf-8

REM Flask 앱 실행
echo 서버를 시작합니다...
echo http://localhost:5000 에서 접속 가능합니다.
echo.
echo Ctrl+C를 눌러 종료하세요.
echo.

python -u app.py

pause