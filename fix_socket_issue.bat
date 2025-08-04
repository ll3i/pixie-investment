@echo off
echo Windows 소켓 문제 해결 스크립트
echo ===================================
echo.

echo 1. Winsock 리셋 중...
netsh winsock reset
echo.

echo 2. IP 설정 리셋 중...
netsh int ip reset
echo.

echo 3. DNS 캐시 플러시 중...
ipconfig /flushdns
echo.

echo 4. 네트워크 어댑터 재시작 중...
netsh interface set interface "Wi-Fi" admin=disable
timeout /t 2 >nul
netsh interface set interface "Wi-Fi" admin=enable

netsh interface set interface "이더넷" admin=disable
timeout /t 2 >nul
netsh interface set interface "이더넷" admin=enable
echo.

echo 완료! 컴퓨터를 재시작하는 것을 권장합니다.
echo.
pause