#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import pandas as pd

# Python path 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Supabase 클라이언트
from supabase import create_client

class OptimizedDataManager:
    def __init__(self):
        self.SUPABASE_URL = os.getenv('SUPABASE_URL')
        self.SUPABASE_KEY = os.getenv('SUPABASE_KEY')
        self.supabase = None
        
        if self.SUPABASE_URL and self.SUPABASE_KEY:
            self.supabase = create_client(self.SUPABASE_URL, self.SUPABASE_KEY)
            print("Supabase 연결 성공!")
        else:
            print("Supabase 연결 실패 - 환경변수 확인 필요")
    
    def get_stock_price_from_supabase(self, stock_code):
        """Supabase에서 특정 종목의 가격 데이터를 직접 조회"""
        try:
            if not self.supabase:
                return None
            
            # 종목코드 6자리로 맞추기
            stock_code = str(stock_code).zfill(6)
            
            # 최근 30일 데이터 조회
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            response = self.supabase.table('kor_stock_prices').select('*').eq(
                'ticker', stock_code
            ).gte(
                'date', thirty_days_ago
            ).order(
                'date', desc=True
            ).execute()
            
            if response.data and len(response.data) > 0:
                print(f"Supabase에서 {stock_code} 종목 {len(response.data)}개 가격 데이터 조회 성공")
                return response.data
            else:
                print(f"Supabase에서 {stock_code} 종목 데이터 없음")
                return None
                
        except Exception as e:
            print(f"Supabase 가격 조회 오류: {e}")
            return None
    
    def get_stock_info_from_supabase(self, stock_code):
        """Supabase에서 종목 정보 조회"""
        try:
            if not self.supabase:
                return None
            
            stock_code = str(stock_code).zfill(6)
            
            # 종목 정보 조회
            response = self.supabase.table('kor_stock_tickers').select('*').eq(
                'ticker', stock_code
            ).order(
                'reference_date', desc=True
            ).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
                
        except Exception as e:
            print(f"Supabase 종목 정보 조회 오류: {e}")
            return None
    
    def get_stock_evaluation_from_supabase(self, stock_code):
        """Supabase에서 종목 평가 정보 조회"""
        try:
            if not self.supabase:
                return None
            
            stock_code = str(stock_code).zfill(6)
            
            # 평가 정보 조회
            response = self.supabase.table('kor_stock_evaluations').select('*').eq(
                'ticker', stock_code
            ).order(
                'reference_date', desc=True
            ).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
                
        except Exception as e:
            print(f"Supabase 평가 정보 조회 오류: {e}")
            return None
    
    def get_latest_valuations_from_supabase(self, stock_code):
        """Supabase에서 최신 밸류에이션 지표 조회"""
        try:
            if not self.supabase:
                return {}
            
            stock_code = str(stock_code).zfill(6)
            
            # 최신 밸류에이션 데이터 조회
            response = self.supabase.table('kor_valuation_metrics').select('*').eq(
                'ticker', stock_code
            ).order(
                'date', desc=True
            ).limit(10).execute()
            
            if response.data:
                # 지표별로 정리
                valuations = {}
                for item in response.data:
                    metric = item.get('metric')
                    value = item.get('value')
                    if metric and value is not None:
                        valuations[metric] = value
                
                return valuations
            
            return {}
                
        except Exception as e:
            print(f"Supabase 밸류에이션 조회 오류: {e}")
            return {}

# 전역 데이터 매니저
optimized_manager = OptimizedDataManager()

@app.route('/api/stock-price', methods=['GET'])
def api_stock_price():
    """종목 가격 조회 API - Supabase 직접 조회"""
    try:
        stock_code = request.args.get('code', '')
        if not stock_code:
            return jsonify({'success': False, 'error': '종목 코드가 필요합니다'})
        
        # Supabase에서 직접 조회
        price_data = optimized_manager.get_stock_price_from_supabase(stock_code)
        stock_info = optimized_manager.get_stock_info_from_supabase(stock_code)
        eval_info = optimized_manager.get_stock_evaluation_from_supabase(stock_code)
        valuations = optimized_manager.get_latest_valuations_from_supabase(stock_code)
        
        if not price_data:
            return jsonify({
                'success': False,
                'error': f'{stock_code} 종목의 가격 데이터를 찾을 수 없습니다'
            })
        
        # 최신 가격 정보
        latest_price = price_data[0]
        
        # 차트 데이터 준비
        chart_data = {
            'labels': [item['date'] for item in reversed(price_data[-30:])],
            'prices': [float(item['close']) for item in reversed(price_data[-30:])]
        }
        
        # 기술적 지표 계산
        closes = [float(item['close']) for item in price_data]
        volumes = [int(item['volume']) for item in price_data]
        
        # 5일, 20일 이동평균
        ma5 = sum(closes[:5]) / 5 if len(closes) >= 5 else closes[0]
        ma20 = sum(closes[:20]) / 20 if len(closes) >= 20 else closes[0]
        
        # 거래량 평균
        avg_volume = sum(volumes[:20]) / 20 if len(volumes) >= 20 else volumes[0]
        
        # 52주 최고/최저
        year_prices = [float(item['close']) for item in price_data[:252]] if len(price_data) > 252 else closes
        high_52w = max(year_prices) if year_prices else latest_price['close']
        low_52w = min(year_prices) if year_prices else latest_price['close']
        
        result = {
            'success': True,
            'stock_name': stock_info.get('name', f'종목 {stock_code}') if stock_info else f'종목 {stock_code}',
            'current_price': float(latest_price['close']),
            'previous_close': float(price_data[1]['close']) if len(price_data) > 1 else float(latest_price['close']),
            'change': float(latest_price['close']) - float(price_data[1]['close']) if len(price_data) > 1 else 0,
            'change_percent': ((float(latest_price['close']) - float(price_data[1]['close'])) / float(price_data[1]['close']) * 100) if len(price_data) > 1 and float(price_data[1]['close']) > 0 else 0,
            'volume': int(latest_price['volume']),
            'high': float(latest_price['high']),
            'low': float(latest_price['low']),
            'open': float(latest_price['open']),
            'market_cap': stock_info.get('market_cap', 0) if stock_info else 0,
            'per': valuations.get('PER', eval_info.get('per', 0) if eval_info else 0),
            'pbr': valuations.get('PBR', eval_info.get('pbr', 0) if eval_info else 0),
            'evaluation_score': eval_info.get('score', 0) if eval_info else 0,
            'evaluation': eval_info.get('evaluation', '') if eval_info else '',
            'reasons': eval_info.get('reasons', '') if eval_info else '',
            'chart_data': chart_data,
            'technical_indicators': {
                'ma5': round(ma5, 2),
                'ma20': round(ma20, 2),
                'volume_avg': int(avg_volume),
                'high_52w': round(high_52w, 2),
                'low_52w': round(low_52w, 2)
            },
            'data_source': 'supabase'
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"API 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'서버 오류: {str(e)}'
        })

@app.route('/api/time_series_prediction', methods=['POST'])
def api_time_series_prediction():
    """시계열 예측 API - Supabase 데이터 사용"""
    try:
        data = request.get_json()
        stock_code = data.get('stock_code')
        days = data.get('days', 7)
        
        if not stock_code:
            return jsonify({
                'success': False,
                'error': '종목코드가 필요합니다'
            })
        
        # Supabase에서 가격 데이터 조회
        price_data = optimized_manager.get_stock_price_from_supabase(stock_code)
        stock_info = optimized_manager.get_stock_info_from_supabase(stock_code)
        
        if not price_data or len(price_data) < 5:
            return jsonify({
                'success': False,
                'error': f'{stock_code} 종목의 충분한 데이터가 없습니다'
            })
        
        # 시계열 분석
        prices = [float(item['close']) for item in price_data]
        dates = [item['date'] for item in price_data]
        
        # 간단한 이동평균 기반 예측
        current_price = prices[0]
        ma5 = sum(prices[:5]) / 5 if len(prices) >= 5 else current_price
        ma20 = sum(prices[:20]) / 20 if len(prices) >= 20 else current_price
        
        # 추세 계산
        trend_short = (current_price - ma5) / ma5 * 100 if ma5 > 0 else 0
        trend_long = (current_price - ma20) / ma20 * 100 if ma20 > 0 else 0
        
        # 변동성 계산
        price_changes = []
        for i in range(1, min(20, len(prices))):
            change = (prices[i-1] - prices[i]) / prices[i] * 100
            price_changes.append(change)
        
        volatility = sum(abs(c) for c in price_changes) / len(price_changes) if price_changes else 5
        
        # 예측 생성
        predictions = []
        for day in range(1, days + 1):
            # 간단한 예측 모델
            trend_effect = (trend_short * 0.3 + trend_long * 0.7) * (day / 10)
            volatility_effect = volatility * (day ** 0.5) / 10
            
            import random
            random_factor = random.uniform(-volatility/10, volatility/10)
            
            predicted_change = trend_effect + random_factor
            predicted_price = current_price * (1 + predicted_change / 100)
            
            # 합리적 범위 제한
            predicted_price = max(current_price * 0.8, min(predicted_price, current_price * 1.2))
            
            predictions.append({
                'day': day,
                'predicted_price': round(predicted_price, 2),
                'confidence': round(0.9 - day * 0.05, 2),
                'price_change_percent': round(predicted_change, 2)
            })
        
        result = {
            'success': True,
            'prediction': {
                'stock_code': stock_code,
                'stock_name': stock_info.get('name', f'종목 {stock_code}') if stock_info else f'종목 {stock_code}',
                'current_price': current_price,
                'predictions': predictions,
                'trend': 'bullish' if trend_short > 2 else 'bearish' if trend_short < -2 else 'neutral',
                'volatility': round(volatility, 2),
                'ma5': round(ma5, 2),
                'ma20': round(ma20, 2),
                'analysis_period': len(prices),
                'data_source': 'supabase',
                'model_type': 'MA-based Prediction'
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"시계열 예측 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': f'서버 오류: {str(e)}'
        })

if __name__ == '__main__':
    print("최적화된 API 서버 시작...")
    app.run(debug=True, port=5001)