"""
뉴스 관련 라우트
뉴스 조회, 키워드 관리, 인사이트 분석 등
"""

from flask import request, jsonify, session
from datetime import datetime, timedelta
import json
from . import news_bp
from services.news_service import NewsService
from services.analysis_service import AnalysisService

# 서비스 초기화
news_service = NewsService()
analysis_service = AnalysisService()

@news_bp.route('', methods=['GET'])
def get_news():
    """뉴스 목록을 조회합니다."""
    try:
        # 파라미터 파싱
        category = request.args.get('category', 'all')
        keyword = request.args.get('keyword', '')
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # 뉴스 조회
        news_data = news_service.get_news(
            category=category,
            keyword=keyword,
            limit=limit,
            offset=offset
        )
        
        return jsonify(news_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/today', methods=['GET'])
def get_today_news():
    """오늘의 뉴스를 조회합니다."""
    try:
        # 카테고리별 뉴스 가져오기
        news_data = news_service.get_today_news()
        
        # 트렌드 키워드 추가
        news_data['trend_keywords'] = news_service.get_trend_keywords()
        
        return jsonify(news_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/portfolio', methods=['GET'])
def get_portfolio_news():
    """포트폴리오 관련 뉴스를 조회합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        # 사용자 포트폴리오 기반 뉴스
        news_data = news_service.get_portfolio_news(user_id)
        
        return jsonify(news_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/watchlist', methods=['POST'])
def get_watchlist_news():
    """관심 종목 관련 뉴스를 조회합니다."""
    try:
        data = request.get_json()
        stock_codes = data.get('stock_codes', [])
        
        if not stock_codes:
            return jsonify({'error': '종목 코드가 필요합니다.'}), 400
        
        news_data = news_service.get_stocks_news(stock_codes)
        
        return jsonify(news_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/popular', methods=['GET'])
def get_popular_news():
    """인기 뉴스를 조회합니다."""
    try:
        hours = request.args.get('hours', 24, type=int)
        limit = request.args.get('limit', 10, type=int)
        
        popular_news = news_service.get_popular_news(hours, limit)
        
        return jsonify({
            'period_hours': hours,
            'news': popular_news
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/sentiment', methods=['GET'])
def get_news_sentiment():
    """뉴스 감성 분석 결과를 조회합니다."""
    try:
        period = request.args.get('period', 'day')  # day, week, month
        category = request.args.get('category', 'all')
        
        sentiment_data = analysis_service.get_news_sentiment(period, category)
        
        return jsonify(sentiment_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/analysis', methods=['POST'])
def analyze_news():
    """뉴스 내용을 AI로 분석합니다."""
    try:
        data = request.get_json()
        news_content = data.get('content')
        analysis_type = data.get('type', 'summary')  # summary, impact, recommendation
        
        if not news_content:
            return jsonify({'error': '뉴스 내용이 필요합니다.'}), 400
        
        analysis_result = analysis_service.analyze_news(news_content, analysis_type)
        
        return jsonify(analysis_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/keywords', methods=['GET', 'POST', 'DELETE'])
def manage_keywords():
    """사용자 관심 키워드를 관리합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        if request.method == 'GET':
            # 키워드 목록 조회
            keywords = news_service.get_user_keywords(user_id)
            return jsonify({'keywords': keywords})
            
        elif request.method == 'POST':
            # 키워드 추가
            data = request.get_json()
            keyword = data.get('keyword')
            
            if not keyword:
                return jsonify({'error': '키워드가 필요합니다.'}), 400
            
            news_service.add_user_keyword(user_id, keyword)
            return jsonify({'success': True, 'message': '키워드가 추가되었습니다.'})
            
        else:  # DELETE
            # 키워드 삭제
            keyword = request.args.get('keyword')
            
            if not keyword:
                return jsonify({'error': '키워드가 필요합니다.'}), 400
            
            news_service.remove_user_keyword(user_id, keyword)
            return jsonify({'success': True, 'message': '키워드가 삭제되었습니다.'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/recommend', methods=['GET'])
def recommend_news():
    """사용자 맞춤 뉴스를 추천합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        # 맞춤 추천 뉴스
        recommendations = news_service.get_personalized_news(user_id)
        
        return jsonify(recommendations)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/scheduler-status', methods=['GET'])
def get_scheduler_status():
    """뉴스 수집 스케줄러 상태를 확인합니다."""
    try:
        status = news_service.get_scheduler_status()
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@news_bp.route('/scheduler-run', methods=['POST'])
def run_scheduler():
    """뉴스 수집을 수동으로 실행합니다."""
    try:
        result = news_service.run_news_collection()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500