@echo off
echo ================================
echo GitHub Push Helper
echo ================================
echo.

echo 설정 변경 중...
git config http.postBuffer 524288000
git config core.compression 0

echo.
echo 푸시 시도 중...
git push -u origin main --verbose

echo.
echo 만약 실패하면 다음을 시도하세요:
echo 1. GitHub Desktop 사용
echo 2. 다음 명령 실행:
echo    git push -u origin main --force
echo.
pause