# Flask Blueprint 리팩토링 가이드

## 🎯 리팩토링 개요

Pixie 웹 애플리케이션을 모놀리식 구조에서 Blueprint 기반 모듈화 구조로 전환했습니다.

### 주요 변경사항

1. **모듈화 구조**
   - 5,910줄의 단일 `app.py` → Blueprint 기반 모듈 분리
   - 기능별 라우트 그룹화
   - 서비스 레이어 도입

2. **새로운 디렉토리 구조**
```
web/
├── app_refactored.py      # 리팩토링된 메인 애플리케이션
├── config.py              # 중앙화된 설정 관리
├── blueprints/            # Blueprint 모듈들
│   ├── __init__.py
│   ├── auth/              # 인증/사용자 관리
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── chat/              # 채팅/AI 상담
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── news/              # 뉴스 관리
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── stock/             # 주식/포트폴리오
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── learning/          # 학습 컨텐츠 (TODO)
│   ├── alerts/            # 알림 관리 (TODO)
│   └── utils/             # 공통 유틸리티
│       └── decorators.py
├── services/              # 비즈니스 로직 서비스
│   ├── __init__.py
│   ├── user_service.py
│   ├── survey_service.py
│   ├── database_service.py
│   └── ... (추가 서비스들)
└── models/                # 데이터 모델 (TODO)
```

## 🚀 실행 방법

### 1. 새로운 애플리케이션 실행
```bash
# 리팩토링된 앱 실행
python app_refactored.py

# 또는 환경 변수 설정 후 실행
export FLASK_ENV=development
python app_refactored.py
```

### 2. 기존 앱과 병행 실행
- 기존 `app.py`는 그대로 유지됨
- 새 구조는 `app_refactored.py`로 별도 실행 가능
- 점진적 마이그레이션 가능

## 📋 Blueprint 구조

### 1. Auth Blueprint (`/auth`)
- `/survey` - 설문조사 페이지
- `/submit_survey` - 설문 제출
- `/survey/result` - 결과 페이지
- `/profile-status` - 프로필 상태
- `/profile` - 프로필 조회

### 2. Chat Blueprint (`/api`)
- `/chat` - AI 채팅 (스트리밍)
- `/chat-stream` - SSE 스트림
- `/chat-history` - 채팅 기록
- `/clear-chat` - 채팅 초기화
- `/export-chat` - 채팅 내보내기

### 3. News Blueprint (`/api/news`)
- `/` - 뉴스 목록
- `/today` - 오늘의 뉴스
- `/portfolio` - 포트폴리오 뉴스
- `/watchlist` - 관심종목 뉴스
- `/popular` - 인기 뉴스
- `/sentiment` - 감성 분석
- `/analysis` - AI 분석
- `/keywords` - 키워드 관리
- `/recommend` - 맞춤 추천

### 4. Stock Blueprint (`/api/stock`)
- `/price` - 주가 조회
- `/info/<code>` - 종목 정보
- `/search` - 종목 검색
- `/list` - 종목 목록
- `/watchlist` - 관심 종목
- `/portfolio/*` - 포트폴리오 관련
- `/time-series-prediction` - 시계열 예측
- `/risk-alerts` - 위험 알림

## 🛠️ 주요 개선사항

### 1. 코드 구조 개선
- **관심사 분리**: 각 기능별로 독립된 모듈
- **재사용성**: 공통 로직을 데코레이터와 서비스로 분리
- **테스트 용이성**: 모듈별 독립 테스트 가능
- **유지보수성**: 기능별 파일 분리로 관리 용이

### 2. 보안 강화
- **데코레이터 기반 인증**: `@require_auth`
- **에러 핸들링**: `@handle_errors`
- **입력 검증**: `@validate_json`
- **Rate Limiting**: `@rate_limit`

### 3. 성능 최적화
- **응답 캐싱**: `@cache_response`
- **요청 로깅**: `@log_request`
- **데이터베이스 연결 풀링** (TODO)
- **비동기 처리** (TODO)

## 🔄 마이그레이션 전략

### Phase 1: 구조 분리 (완료)
- [x] Blueprint 디렉토리 구조 생성
- [x] 인증/사용자 Blueprint 분리
- [x] 채팅/AI Blueprint 분리
- [x] 뉴스 Blueprint 분리
- [x] 주식/포트폴리오 Blueprint 분리
- [x] 공통 유틸리티 분리
- [x] 설정 파일 중앙화

### Phase 2: 서비스 레이어 (진행중)
- [x] UserService 구현
- [x] SurveyService 구현
- [x] DatabaseService 구현
- [ ] ChatService 구현
- [ ] NewsService 구현
- [ ] StockService 구현
- [ ] PortfolioService 구현

### Phase 3: 추가 개선 (TODO)
- [ ] 학습 Blueprint 분리
- [ ] 알림 Blueprint 분리
- [ ] 데이터 모델 정의
- [ ] API 버전관리 도입
- [ ] OpenAPI 문서화
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 작성

## ⚠️ 주의사항

1. **호환성 유지**
   - 기존 API 엔드포인트는 동일하게 유지
   - 프론트엔드 수정 불필요
   - 점진적 마이그레이션 가능

2. **환경 변수**
   - 기존 `.env` 파일 그대로 사용
   - 새로운 설정은 `config.py`에서 관리

3. **데이터베이스**
   - 기존 SQLite 데이터베이스 호환
   - 스키마 변경 없음

## 🚧 다음 단계

1. **서비스 레이어 완성**
   ```bash
   /implement remaining-services --type service
   ```

2. **테스트 작성**
   ```bash
   /implement unit-tests --framework pytest
   ```

3. **API 문서화**
   ```bash
   /implement api-documentation --type openapi
   ```

4. **성능 최적화**
   ```bash
   /improve performance --focus database-queries
   ```

5. **프로덕션 준비**
   ```bash
   /implement production-ready --focus security
   ```

## 📚 참고 문서

- [Flask Blueprints 공식 문서](https://flask.palletsprojects.com/en/2.0.x/blueprints/)
- [Flask 애플리케이션 팩토리 패턴](https://flask.palletsprojects.com/en/2.0.x/patterns/appfactories/)
- [Flask 테스팅 가이드](https://flask.palletsprojects.com/en/2.0.x/testing/)