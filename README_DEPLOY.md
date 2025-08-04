# Pixie 배포 가이드

## Vercel 배포 방법 (가장 쉬운 방법)

### 1. 사전 준비
- GitHub 계정
- Vercel 계정 (GitHub로 가입 가능)

### 2. 배포 단계

#### Step 1: GitHub에 코드 업로드
```bash
# web 폴더로 이동
cd web

# Git 초기화 (처음인 경우)
git init

# 파일 추가
git add .

# 커밋
git commit -m "Initial Pixie deployment"

# GitHub 리포지토리 생성 후
git remote add origin https://github.com/YOUR_USERNAME/pixie-investment.git
git push -u origin main
```

#### Step 2: Vercel에서 배포
1. [Vercel](https://vercel.com) 접속
2. "New Project" 클릭
3. GitHub 리포지토리 선택
4. 프로젝트 설정:
   - Framework Preset: Other
   - Root Directory: `web` (web 폴더 선택)
   - Build Command: 비워두기
   - Output Directory: 비워두기

5. 환경 변수 설정:
   - `SECRET_KEY`: 32자 이상의 랜덤 문자열

6. "Deploy" 클릭

### 3. 배포 확인
- 배포 완료 후 제공되는 URL로 접속
- 예: `https://pixie-investment.vercel.app`

### 4. 도메인 설정 (선택사항)
- Vercel 대시보드에서 Settings > Domains
- 커스텀 도메인 추가 가능

## 로컬 테스트

```bash
# 필요한 패키지 설치
pip install -r requirements_deploy.txt

# 앱 실행
python app_vercel.py

# 브라우저에서 확인
# http://localhost:5000
```

## 주요 파일 설명

- `app_vercel.py`: 배포용 간소화된 Flask 앱
- `vercel.json`: Vercel 배포 설정
- `requirements_deploy.txt`: 최소 필요 패키지
- `templates/`: HTML 템플릿 파일들
- `static/`: CSS, JS, 이미지 파일들

## 문제 해결

### 배포 실패 시
1. `vercel.json` 파일 확인
2. `requirements_deploy.txt` 파일 확인
3. Vercel 로그 확인

### 페이지가 안 보일 때
1. 템플릿 파일 경로 확인
2. static 파일 경로 확인
3. Flask 라우트 확인

## 대체 배포 옵션

### Render.com (무료)
1. [Render](https://render.com) 가입
2. New > Web Service
3. GitHub 연결
4. 자동 배포 설정

### PythonAnywhere (무료)
1. [PythonAnywhere](https://www.pythonanywhere.com) 가입
2. 파일 업로드
3. Web 탭에서 Flask 앱 설정

### Railway (간단한 배포)
1. [Railway](https://railway.app) 가입
2. GitHub 연결
3. 자동 배포

## 지원
- 문제 발생 시 GitHub Issues에 문의
- Pixie 팀 이메일: support@pixie.ai (예시)