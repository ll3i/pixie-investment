"""
주식 관련 라우트
주가 조회, 포트폴리오 관리, 관심종목 등
"""

from flask import request, jsonify, session
from datetime import datetime
import json
from . import stock_bp
from services.stock_service import StockService
from services.portfolio_service import PortfolioService

# 서비스 초기화
stock_service = StockService()
portfolio_service = PortfolioService()

@stock_bp.route('/price', methods=['GET'])
def get_stock_price():
    """주식 가격 정보를 조회합니다."""
    try:
        stock_code = request.args.get('code')
        period = request.args.get('period', '1M')  # 1D, 1W, 1M, 3M, 1Y
        
        if not stock_code:
            return jsonify({'error': '종목 코드가 필요합니다.'}), 400
        
        price_data = stock_service.get_stock_price(stock_code, period)
        
        return jsonify(price_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/info/<stock_code>')
def get_stock_info(stock_code):
    """주식 상세 정보를 조회합니다."""
    try:
        stock_info = stock_service.get_stock_info(stock_code)
        
        if not stock_info:
            return jsonify({'error': '종목 정보를 찾을 수 없습니다.'}), 404
        
        return jsonify(stock_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/search')
def search_stocks():
    """주식을 검색합니다."""
    try:
        query = request.args.get('q', '')
        market = request.args.get('market', 'all')  # all, kospi, kosdaq, nasdaq
        limit = request.args.get('limit', 20, type=int)
        
        if not query:
            return jsonify({'error': '검색어가 필요합니다.'}), 400
        
        results = stock_service.search_stocks(query, market, limit)
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/list', methods=['GET'])
def list_stocks():
    """주식 목록을 조회합니다."""
    try:
        market = request.args.get('market', 'all')
        sector = request.args.get('sector')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        stocks = stock_service.list_stocks(
            market=market,
            sector=sector,
            limit=limit,
            offset=offset
        )
        
        return jsonify(stocks)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/watchlist', methods=['GET'])
def get_watchlist():
    """관심 종목 목록을 조회합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        watchlist = stock_service.get_watchlist(user_id)
        
        return jsonify({
            'watchlist': watchlist,
            'count': len(watchlist)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/watchlist', methods=['POST'])
def add_to_watchlist():
    """관심 종목을 추가합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        data = request.get_json()
        stock_code = data.get('stock_code')
        stock_name = data.get('stock_name')
        market = data.get('market', 'KOSPI')
        
        if not stock_code:
            return jsonify({'error': '종목 코드가 필요합니다.'}), 400
        
        result = stock_service.add_to_watchlist(
            user_id, stock_code, stock_name, market
        )
        
        if result:
            return jsonify({'success': True, 'message': '관심 종목에 추가되었습니다.'})
        else:
            return jsonify({'error': '이미 관심 종목에 있습니다.'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/watchlist/<stock_code>', methods=['DELETE'])
def remove_from_watchlist(stock_code):
    """관심 종목을 삭제합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        result = stock_service.remove_from_watchlist(user_id, stock_code)
        
        if result:
            return jsonify({'success': True, 'message': '관심 종목에서 삭제되었습니다.'})
        else:
            return jsonify({'error': '관심 종목을 찾을 수 없습니다.'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/portfolio/recommendations')
def get_portfolio_recommendations():
    """포트폴리오 추천을 조회합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        # 사용자 프로필 기반 추천
        recommendations = portfolio_service.get_recommendations(user_id)
        
        return jsonify(recommendations)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/portfolio/analysis', methods=['POST'])
def analyze_portfolio():
    """포트폴리오를 분석합니다."""
    try:
        data = request.get_json()
        portfolio = data.get('portfolio', [])
        
        if not portfolio:
            return jsonify({'error': '포트폴리오 데이터가 필요합니다.'}), 400
        
        analysis = portfolio_service.analyze_portfolio(portfolio)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/portfolio/history', methods=['GET'])
def get_portfolio_history():
    """포트폴리오 추천 기록을 조회합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        limit = request.args.get('limit', 10, type=int)
        history = portfolio_service.get_recommendation_history(user_id, limit)
        
        return jsonify({
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/time-series-prediction', methods=['POST'])
def predict_time_series():
    """시계열 예측을 수행합니다."""
    try:
        data = request.get_json()
        stock_code = data.get('stock_code')
        days = data.get('days', 30)
        
        if not stock_code:
            return jsonify({'error': '종목 코드가 필요합니다.'}), 400
        
        prediction = stock_service.predict_stock_price(stock_code, days)
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@stock_bp.route('/risk-alerts')
def get_risk_alerts():
    """투자 위험 알림을 조회합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        alerts = stock_service.get_risk_alerts(user_id)
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500