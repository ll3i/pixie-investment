@echo off
echo TensorFlow 없이 Flask 앱 실행...
echo.

REM TensorFlow 관련 경고 비활성화
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0

REM Python 버퍼링 비활성화
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM Flask 설정
set FLASK_APP=app.py
set FLASK_ENV=production

echo 서버를 시작합니다...
echo http://localhost:5000 에서 접속 가능합니다.
echo.

python simple_run.py

pause