@echo off
echo ================================
echo Pixie GitHub Upload Script
echo ================================
echo.

echo Step 1: Git 초기화
git init

echo.
echo Step 2: 파일 추가
git add .

echo.
echo Step 3: 커밋 생성
git commit -m "Initial Pixie deployment - AI Investment Advisor"

echo.
echo Step 4: GitHub 원격 저장소 추가
echo GitHub에서 'pixie-investment' 리포지토리를 먼저 생성하세요!
echo.
set /p username="GitHub 사용자명을 입력하세요: "
git remote add origin https://github.com/%username%/pixie-investment.git

echo.
echo Step 5: 메인 브랜치로 변경
git branch -M main

echo.
echo Step 6: GitHub에 푸시
git push -u origin main

echo.
echo ================================
echo 업로드 완료!
echo ================================
echo.
echo Vercel 배포 단계:
echo 1. https://vercel.com 접속
echo 2. New Project 클릭
echo 3. GitHub 리포지토리 선택
echo 4. Root Directory를 'web'으로 설정
echo 5. Deploy 클릭
echo.
pause