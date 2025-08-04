#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MINERVA 투자 분석 웹 애플리케이션 (V2)
- 서비스 알고리즘 흐름도에 맞게 재구성된 버전
- 사용자 성향 분석, 금융 데이터 처리, LLM 서비스 통합
"""

import os
import sys
import json
import time
import uuid
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from datetime import datetime
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 개발 환경을 위한 기본 환경변수 설정
if not os.environ.get("FLASK_SECRET_KEY"):
    os.environ["FLASK_SECRET_KEY"] = "minerva_investment_chatbot_secure_key_2024_very_long_secret"

if not os.environ.get("FLASK_ENV"):
    os.environ["FLASK_ENV"] = "development"

# 현재 디렉토리를 파이썬 경로에 추가하여 src 모듈을 import할 수 있게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# src 모듈 임포트
try:
    from src.user_profile_analyzer import UserProfileAnalyzer
    from src.financial_data_processor import FinancialDataProcessor
    from src.llm_service import LLMService
    from src.memory_manager import MemoryManager
    from src.investment_advisor import InvestmentAdvisor
    from src.simplified_portfolio_prediction import extract_portfolio_tickers, analyze_portfolio_with_user_profile
    modules_imported = True
    print("모듈 임포트 성공")
except ImportError as e:
    print(f"모듈 임포트 오류: {e}")
    modules_imported = False

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "minerva_investment_advisor")

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
IMAGE_DIR = os.path.join(app.static_folder, 'images')
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.json")
ANALYSIS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results.json")

# 디렉토리 초기화
os.makedirs(IMAGE_DIR, exist_ok=True)

# 전역 변수 초기화
advisor = None
profile_analyzer = None
financial_processor = None
memory_manager = None
llm_service = None

# 시스템 초기화 함수
def initialize_system():
    global advisor, profile_analyzer, financial_processor, memory_manager, llm_service
    
    if not modules_imported:
        print("모듈 임포트 실패로 시스템을 초기화할 수 없습니다.")
        return False
    
    try:
        # API 유형 결정
        api_type = "openai" if os.environ.get('OPENAI_API_KEY') else "clova"
        if not os.environ.get('OPENAI_API_KEY') and not os.environ.get('CLOVA_API_KEY'):
            print("경고: API 키가 설정되지 않았습니다.")
            api_type = "simulation"
        
        # 각 모듈 초기화
        profile_analyzer = UserProfileAnalyzer()
        financial_processor = FinancialDataProcessor()
        memory_manager = MemoryManager()
        llm_service = LLMService(api_type=api_type)
        
        # 통합 시스템 초기화
        advisor = InvestmentAdvisor(api_type=api_type)
        
        print("시스템 초기화 완료")
        return True
    except Exception as e:
        print(f"시스템 초기화 오류: {e}")
        return False

# 시스템 초기화 시도
system_initialized = initialize_system()

# 세션 관리 함수
def get_or_create_session_id():
    """세션 ID 가져오기 또는 생성"""
    if 'session_id' not in session:
        session['session_id'] = f"user_{uuid.uuid4().hex[:8]}"
    return session['session_id']

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/survey')
def survey():
    """설문조사 페이지를 렌더링합니다."""
    return render_template('survey.html')

@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    """설문 조사 결과를 제출하고 결과를 분석합니다."""
    try:
        data = request.get_json()
        answers = data.get('answers', [])
        
        # 세션 ID 가져오기
        session_id = get_or_create_session_id()
        
        if system_initialized and advisor:
            # 어드바이저를 통한 설문 분석
            advisor.set_session_id(session_id)
            result = advisor.analyze_survey_responses(answers)
            
            # 분석 결과를 파일에 저장
            try:
                with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                print(f"분석 결과가 저장되었습니다: {ANALYSIS_FILE}")
            except Exception as e:
                print(f"분석 결과 저장 중 오류 발생: {e}")
        else:
            # 폴백: 기존 방식으로 설문 분석
            result = analyze_survey_responses_fallback(answers)
            
            # 분석 결과를 파일에 저장
            try:
                with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                print(f"분석 결과가 저장되었습니다: {ANALYSIS_FILE}")
            except Exception as e:
                print(f"분석 결과 저장 중 오류 발생: {e}")
        
        return jsonify(result)
    except Exception as e:
        print(f"설문 제출 오류: {e}")
        return jsonify({'error': '설문 처리 중 오류가 발생했습니다.'}), 500

def analyze_survey_responses_fallback(answers):
    """설문조사 응답을 분석하여 투자 성향 점수를 계산합니다. (폴백 함수)"""
    # 기본 점수 설정
    scores = {
        'risk_tolerance': 50,
        'investment_time_horizon': 50,
        'financial_goal_orientation': 50,
        'information_processing_style': 50,
        'investment_fear': 50,
        'investment_confidence': 50
    }
    
    # 간단한 분석 로직
    if len(answers) > 0:
        answer = answers[0].lower() if isinstance(answers[0], str) else answers[0]['answer'].lower()
        if "안전" in answer or "안정" in answer:
            scores['risk_tolerance'] = 30
        elif "위험" in answer or "수익" in answer:
            scores['risk_tolerance'] = 70
    
    if len(answers) > 2:
        answer = answers[2].lower() if isinstance(answers[2], str) else answers[2]['answer'].lower()
        if "단기" in answer:
            scores['investment_time_horizon'] = 30
        elif "장기" in answer or "노후" in answer:
            scores['investment_time_horizon'] = 70
    
    # 분석 결과 생성
    detailed_analysis = {
        'risk_tolerance_analysis': "위험 감수성이 중간 수준으로, 적절한 위험을 감수하면서 수익을 추구하는 균형잡힌 투자를 선호합니다." if scores['risk_tolerance'] == 50 else
                                  ("위험 감수성이 낮아 안전 자산 위주의 투자를 선호합니다." if scores['risk_tolerance'] < 50 else
                                   "위험 감수성이 높아 높은 수익을 위해 위험을 감수할 의향이 있습니다."),
        'investment_time_horizon_analysis': "중기 투자 성향으로, 1-5년 정도의 투자 기간을 고려합니다." if scores['investment_time_horizon'] == 50 else
                                          ("단기 투자 성향을 보이며, 1년 미만의 투자 기간을 선호합니다." if scores['investment_time_horizon'] < 50 else
                                           "장기 투자 성향으로, 5년 이상의 장기적 관점에서 투자를 계획합니다."),
        'financial_goal_orientation_analysis': "균형 잡힌 재무 목표를 가지고 있으며, 중기적인 자산 증식을 목표로 합니다.",
        'information_processing_style_analysis': "균형 잡힌 정보 처리 스타일로, 직관과 분석을 모두 활용하여 투자 결정을 내립니다.",
        'investment_confidence_analysis': "투자에 대한 중간 수준의 자신감을 가지고 있으며, 기본적인 투자 개념에 익숙합니다."
    }
    
    overall_analysis = "귀하는 균형 잡힌 투자 성향을 가지고 있으며, 적절한 위험과 수익의 균형을 추구합니다. 중기적인 투자 기간을 선호하며, 안정적인 자산과 성장 자산을 적절히 혼합한 포트폴리오가 적합할 것입니다."
    
    return {
        'scores': scores,
        'overall_analysis': overall_analysis,
        'detailed_analysis': detailed_analysis
    }

@app.route('/result')
def result():
    """결과 페이지를 렌더링합니다."""
    # 분석 결과 로드
    analysis_result = load_analysis_result()
    
    if not analysis_result:
        # 분석 결과가 없으면 설문조사 페이지로 리디렉션
        return redirect(url_for('survey'))
    
    # 세션 ID 가져오기
    session_id = get_or_create_session_id()
    
    # 시스템이 초기화되었고 어드바이저가 있으면 세션 ID 설정
    if system_initialized and advisor:
        advisor.set_session_id(session_id)
    
    return render_template('result.html', 
                          analysis=analysis_result,
                          scores=analysis_result.get('scores', {}),
                          overall_analysis=analysis_result.get('overall_analysis', ''),
                          detailed_analysis=analysis_result.get('detailed_analysis', {}))

@app.route('/minerva')
def minerva():
    """MINERVA 페이지를 렌더링합니다."""
    # 세션 ID 가져오기
    session_id = get_or_create_session_id()
    
    # 시스템이 초기화되었고 어드바이저가 있으면 세션 ID 설정
    if system_initialized and advisor:
        advisor.set_session_id(session_id)
    
    return render_template('minerva.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """채팅 API 엔드포인트"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message.strip():
            return jsonify({'error': '메시지가 비어 있습니다.'}), 400
        
        # 세션 ID 가져오기
        session_id = get_or_create_session_id()
        
        # 분석 결과 로드
        analysis_result = load_analysis_result()
        print(f"분석 결과 로드: {analysis_result}")
        
        if system_initialized and advisor:
            # 어드바이저를 통한 응답 생성
            advisor.set_session_id(session_id)
            print(f"어드바이저 세션 ID 설정: {session_id}")
            
            # 디버깅을 위한 콜백 설정
            def status_callback(status):
                print(f"상태 업데이트: {status}")
            
            def response_callback(agent, response):
                print(f"{agent} 응답: {response[:100]}...")
            
            advisor.set_callbacks(status_callback, response_callback)
            
            response = advisor.chat(message)
            print(f"최종 응답: {response[:100]}...")
            
            # 최종 응답 반환
            return jsonify({'response': response})
        else:
            # 폴백: 기존 방식으로 응답 생성
            print("시스템이 초기화되지 않아 폴백 사용")
            response = fallback_response(message, analysis_result)
            return jsonify({'response': response})
    except Exception as e:
        print(f"채팅 API 오류: {e}")
        return jsonify({'error': '응답 생성 중 오류가 발생했습니다.'}), 500

@app.route('/api/predictions')
def get_predictions():
    """포트폴리오 예측 결과 API 엔드포인트"""
    try:
        # 이미지 파일 목록 가져오기
        prediction_images = []
        for file in os.listdir(IMAGE_DIR):
            if file.endswith('_prediction.png'):
                ticker = file.split('_')[0]
                prediction_images.append({
                    'ticker': ticker,
                    'name': get_ticker_name(ticker),
                    'image_url': f'/static/images/{file}'
                })
        
        # 최신 순으로 정렬
        prediction_images.sort(key=lambda x: os.path.getmtime(os.path.join(IMAGE_DIR, x['image_url'].split('/')[-1])), reverse=True)
        
        return jsonify({'predictions': prediction_images})
    except Exception as e:
        print(f"예측 결과 API 오류: {e}")
        return jsonify({'error': '예측 결과를 불러오는 중 오류가 발생했습니다.'}), 500

@app.route('/api/chat-history')
def get_chat_history():
    """채팅 기록 API 엔드포인트"""
    try:
        # 세션 ID 가져오기
        session_id = get_or_create_session_id()
        
        if system_initialized and advisor:
            # 어드바이저를 통한 채팅 기록 가져오기
            advisor.set_session_id(session_id)
            history = advisor.get_chat_history()
        else:
            # 폴백: 기존 방식으로 채팅 기록 가져오기
            history = load_chat_history()
        
        return jsonify({'history': history})
    except Exception as e:
        print(f"채팅 기록 API 오류: {e}")
        return jsonify({'error': '채팅 기록을 불러오는 중 오류가 발생했습니다.'}), 500

def load_analysis_result():
    """분석 결과를 파일에서 로드합니다."""
    try:
        # 세션 ID 가져오기
        session_id = get_or_create_session_id()
        
        if system_initialized and profile_analyzer:
            # 프로필 분석기를 통한 분석 결과 로드
            return profile_analyzer.load_analysis_result(session_id)
        else:
            # 폴백: 파일에서 직접 로드
            if os.path.exists(ANALYSIS_FILE):
                with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
    except Exception as e:
        print(f"분석 결과 로드 오류: {e}")
        return None

def fallback_response(message, analysis_result):
    """LLM 서비스 실패 시 폴백 응답 생성"""
    # 기본 응답
    default_responses = [
        "죄송합니다만, 현재 AI 서비스에 일시적인 문제가 있습니다. 잠시 후 다시 시도해주세요.",
        "현재 서버가 혼잡합니다. 잠시 후 다시 질문해주시겠어요?",
        "질문을 처리하는 중에 오류가 발생했습니다. 다른 질문을 해보시겠어요?",
        "죄송합니다. 현재 이 질문에 대한 답변을 생성할 수 없습니다. 다른 주제로 질문해주세요."
    ]
    
    # 투자 관련 질문인지 확인
    investment_keywords = ["투자", "주식", "펀드", "etf", "포트폴리오", "자산", "배분", "수익률", "위험", "금융", "재테크"]
    is_investment_question = any(keyword in message.lower() for keyword in investment_keywords)
    
    # 설문 분석 요청인지 확인 - 키워드 확장
    survey_keywords = ["투자성향", "설문", "분석", "결과", "알려줘", "보여줘", "응답"]
    is_survey_analysis_request = any(keyword in message for keyword in survey_keywords) and len([k for k in survey_keywords if k in message]) >= 2
    
    # 설문 완료 여부 확인
    has_completed_survey = analysis_result is not None and 'detailed_analysis' in analysis_result and len(analysis_result.get('detailed_analysis', {})) > 0
    
    # 디버깅 로그 추가
    print(f"메시지: {message}")
    print(f"투자 관련 질문: {is_investment_question}")
    print(f"설문 분석 요청: {is_survey_analysis_request}")
    print(f"설문 완료 여부: {has_completed_survey}")
    print(f"분석 결과: {analysis_result is not None}")
    
    # 설문 미완료 상태에서 투자 관련 질문이나 분석 요청이 들어온 경우
    if (is_investment_question or is_survey_analysis_request) and not has_completed_survey:
        return "투자 성향 설문을 아직 완료하지 않으셨습니다. 맞춤형 투자 조언을 받기 위해서는 먼저 설문조사를 완료해주세요.\n\n설문조사 페이지로 이동하려면 브라우저에서 '/survey' 페이지를 방문해주세요. 설문 완료 후 더 정확한 투자 조언을 제공해드리겠습니다."
    
    # 투자성향설문 분석 요청 처리
    if is_survey_analysis_request and has_completed_survey:
        detailed = analysis_result.get('detailed_analysis', {})
        scores = analysis_result.get('scores', {})
        
        response = "귀하의 투자 성향 분석 결과입니다:\n\n"
        
        # 위험 감수성
        response += f"1. 위험 감수성: {scores.get('risk_tolerance', 0)}점\n"
        response += f"   {detailed.get('risk_tolerance_analysis', '정보 없음')}\n\n"
        
        # 투자 시간 범위
        response += f"2. 투자 시간 범위: {scores.get('investment_time_horizon', 0)}점\n"
        response += f"   {detailed.get('investment_time_horizon_analysis', '정보 없음')}\n\n"
        
        # 재무 목표 지향성
        response += f"3. 재무 목표 지향성: {scores.get('financial_goal_orientation', 0)}점\n"
        response += f"   {detailed.get('financial_goal_orientation_analysis', '정보 없음')}\n\n"
        
        # 정보 처리 스타일
        response += f"4. 정보 처리 스타일: {scores.get('information_processing_style', 0)}점\n"
        response += f"   {detailed.get('information_processing_style_analysis', '정보 없음')}\n\n"
        
        # 투자 두려움
        response += f"5. 투자 두려움: {scores.get('investment_fear', 0)}점\n"
        response += f"   {detailed.get('investment_fear_analysis', '정보 없음')}\n\n"
        
        # 투자 자신감
        response += f"6. 투자 자신감: {scores.get('investment_confidence', 0)}점\n"
        response += f"   {detailed.get('investment_confidence_analysis', '정보 없음')}\n\n"
        
        # 위험 감수성과 투자 시간 범위에 따른 맞춤형 조언
        if scores.get('risk_tolerance', 50) < 30:
            if scores.get('investment_time_horizon', 50) < 30:
                response += "이러한 투자 성향을 고려할 때, 귀하에게는 안전 자산 위주의 단기 포트폴리오가 적합합니다. 낮은 위험 감수성과 단기 투자 성향을 고려하여 MMF, 국공채 ETF, 우량 회사채 등 안정적인 수익을 추구하는 투자 전략을 추천드립니다."
            else:
                response += "이러한 투자 성향을 고려할 때, 귀하에게는 안전 자산 위주의 장기 포트폴리오가 적합합니다. 낮은 위험 감수성을 고려하되, 장기 투자를 통해 복리 효과를 누릴 수 있는 우량 배당주, 채권형 ETF 위주의 투자 전략을 추천드립니다."
        else:
            if scores.get('investment_time_horizon', 50) < 30:
                response += "이러한 투자 성향을 고려할 때, 귀하에게는 중간 위험의 단기 포트폴리오가 적합합니다. 적절한 위험 감수성과 단기 투자 성향을 고려하여 섹터 ETF, 우량 성장주 등을 포함한 균형 잡힌 투자 전략을 추천드립니다."
            else:
                response += "이러한 투자 성향을 고려할 때, 귀하에게는 성장 자산과 안전 자산을 균형 있게 배분한 포트폴리오가 적합합니다. 장기 투자 성향을 활용하여 인덱스 ETF, 우량 성장주, 글로벌 자산 등 다양한 자산군에 분산 투자하는 전략을 추천드립니다."
        
        return response
    
    # 키워드 기반 간단한 응답
    if "안녕" in message or "반가" in message:
        if not has_completed_survey:
            return "안녕하세요! MINERVA 투자 상담 시스템입니다. 맞춤형 투자 조언을 받기 위해서는 먼저 투자 성향 설문조사를 완료해주세요. 설문조사 페이지로 이동하려면 브라우저에서 '/survey' 페이지를 방문해주세요."
        else:
            return "안녕하세요! MINERVA 투자 상담 시스템입니다. 투자에 관한 질문이 있으신가요? 이미 투자 성향 설문을 완료하셨으니 맞춤형 조언을 제공해드릴 수 있습니다."
    
    elif "코스피" in message or "주가" in message:
        return "현재 코스피 지수는 변동성이 있지만 전반적으로 안정세를 유지하고 있습니다. 특정 종목에 관심이 있으신가요?"
    
    elif "추천" in message or "포트폴리오" in message:
        if has_completed_survey:
            risk = analysis_result['detailed_analysis'].get('risk_tolerance_analysis', '')
            if "낮" in risk:
                return "귀하의 위험 성향을 고려할 때, 안정적인 ETF와 우량주 위주의 포트폴리오를 추천드립니다. 채권형 ETF 60%, 대형 우량주 30%, 현금성 자산 10% 정도의 배분이 적합할 것 같습니다."
            elif "높" in risk:
                return "귀하의 위험 감수 성향을 고려할 때, 성장주와 섹터 ETF에 집중한 포트폴리오가 적합할 것 같습니다. 기술주 40%, 성장형 ETF 40%, 우량주 20% 정도의 배분을 고려해보세요."
            else:
                return "귀하의 균형 잡힌 투자 성향을 고려할 때, 안정성과 성장성을 모두 갖춘 포트폴리오가 적합합니다. 인덱스 ETF 40%, 우량 배당주 30%, 성장주 20%, 현금성 자산 10% 정도의 배분을 추천드립니다."
        else:
            return "투자 포트폴리오 구성은 개인의 위험 성향, 투자 기간, 재무 목표에 따라 달라집니다. 맞춤형 추천을 받기 위해서는 먼저 투자 성향 설문조사를 완료해주세요. 설문조사 페이지로 이동하려면 브라우저에서 '/survey' 페이지를 방문해주세요."
    
    elif "etf" in message.lower() or "인덱스" in message:
        return "ETF는 분산 투자에 좋은 수단입니다. 국내 ETF 중에서는 KODEX 200, TIGER 200 등의 대형주 ETF와 ARIRANG 중형주, KBSTAR 배당 등 다양한 옵션이 있습니다. 특정 섹터에 관심이 있으신가요?"
    
    elif "배당" in message or "dividend" in message.lower():
        return "배당주 투자는 안정적인 현금흐름을 원하는 투자자에게 적합합니다. 국내 주요 배당주로는 삼성전자, KT&G, 한국전력, 현대차 등이 있습니다. 배당수익률과 배당성장률을 함께 고려하는 것이 중요합니다."
    
    # 임의 선택
    import random
    return random.choice(default_responses)

def get_ticker_name(code):
    """종목 코드로 종목명 조회"""
    if system_initialized and financial_processor:
        # 금융 데이터 프로세서를 통한 종목명 조회
        company_name = financial_processor._get_company_name(code)
        if company_name != f"종목({code})":
            return company_name
    
    # 폴백: 내장 맵에서 조회
    ticker_map = {
        '005930': '삼성전자',
        '035420': '네이버',
        '000660': 'SK하이닉스',
        '035720': '카카오',
        '051910': 'LG화학',
        '005380': '현대차',
        '068270': '셀트리온',
        '207940': '삼성바이오로직스',
        '035720': '카카오',
        '005490': 'POSCO홀딩스',
        '012330': '현대모비스',
        '055550': '신한지주',
        '105560': 'KB금융',
        '066570': 'LG전자',
        '096770': 'SK이노베이션',
        '017670': 'SK텔레콤',
        '015760': '한국전력',
        '034730': 'SK',
        '018260': '삼성에스디에스',
        '003550': 'LG',
        '036570': '엔씨소프트',
        '090430': '아모레퍼시픽',
        '000270': '기아',
        '009150': '삼성전기',
        '030200': 'KT',
        '086790': '하나금융지주',
        '010130': '고려아연',
        '011170': '롯데케미칼',
        '009540': '한국조선해양',
        '316140': '우리금융지주',
        '047810': '한국항공우주',
        '251270': '넷마블',
        '032830': '삼성생명',
        '010950': 'S-Oil',
        '034220': 'LG디스플레이',
        '024110': '기업은행',
        '011200': 'HMM',
        '006400': '삼성SDI',
        '028260': '삼성물산',
        '097950': 'CJ제일제당',
        '003490': '대한항공',
        '012450': '한화에어로스페이스',
        '008770': '호텔신라',
        '010140': '삼성중공업',
        '016360': '삼성증권',
        '139480': '이마트',
        '032640': 'LG유플러스',
        '021240': '코웨이',
        '011780': '금호석유',
        '006800': '미래에셋증권',
        '071050': '한국금융지주',
        '000720': '현대건설',
        '011070': 'LG이노텍',
        '128940': '한미약품',
        '180640': '한진칼',
        '000810': '삼성화재',
        '004020': '현대제철',
        '010620': '현대미포조선',
        '035250': '강원랜드',
        '267250': 'HD현대',
        '004990': 'DB하이텍',
        '138040': '메리츠금융지주',
        '001040': 'CJ',
        '010060': 'OCI',
        '011790': 'SKC',
        '042670': '두산인프라코어',
        '009830': '한화솔루션',
        '000100': '유한양행',
        '282330': 'BGF리테일',
        '008930': '한미사이언스',
        '012750': 'S-Oil우',
        '030000': '제일기획',
        '192820': '코스맥스',
        '006260': 'LS',
        '005830': 'DB손해보험',
        '004170': '신세계',
        '069960': '현대백화점',
        '000120': '대한전선',
        '005940': 'NH투자증권',
        '069500': 'KODEX 200',
        '091990': '셀트리온헬스케어',
        '247540': '에코프로비엠',
        '293490': '카카오게임즈',
        '196170': '알테오젠',
        '066970': '엘앤에프'
    }
    
    return ticker_map.get(code, f"종목({code})")

def add_to_chat_history(speaker, content):
    """채팅 기록에 메시지 추가"""
    try:
        history = load_chat_history()
        if not history:
            history = []
        
        history.append({
            'speaker': speaker,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        save_chat_history(history)
    except Exception as e:
        print(f"채팅 기록 추가 오류: {e}")

def load_chat_history():
    """채팅 기록 로드"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"채팅 기록 로드 오류: {e}")
        return []

def save_chat_history(history):
    """채팅 기록 저장"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"채팅 기록 저장 오류: {e}")

# 애플리케이션 실행 시 추가 설정
if __name__ == '__main__':
    try:
        # 기본 포트 설정
        port = int(os.environ.get('PORT', 5000))
        
        # 디버그 모드 설정
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        
        # 애플리케이션 실행
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        print(f"애플리케이션 실행 오류: {e}") 