"""
공통 데코레이터
인증, 에러 처리, 로깅 등
"""

from functools import wraps
from flask import jsonify, session, request
import time
import logging

logger = logging.getLogger(__name__)

def require_auth(f):
    """인증이 필요한 엔드포인트 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        return f(*args, **kwargs)
    return decorated_function

def handle_errors(f):
    """에러 처리 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"ValueError in {f.__name__}: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except KeyError as e:
            logger.warning(f"KeyError in {f.__name__}: {str(e)}")
            return jsonify({'error': f'필수 필드가 없습니다: {str(e)}'}), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({'error': '서버 오류가 발생했습니다.'}), 500
    return decorated_function

def log_request(f):
    """요청 로깅 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        # 요청 정보 로깅
        logger.info(f"Request: {request.method} {request.path} - User: {session.get('user_id', 'anonymous')}")
        
        # 함수 실행
        result = f(*args, **kwargs)
        
        # 응답 시간 로깅
        duration = time.time() - start_time
        logger.info(f"Response: {request.path} - Duration: {duration:.3f}s")
        
        return result
    return decorated_function

def validate_json(required_fields=None):
    """JSON 요청 검증 데코레이터"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({'error': 'JSON 형식의 요청이 필요합니다.'}), 400
            
            data = request.get_json()
            if not data:
                return jsonify({'error': '요청 데이터가 없습니다.'}), 400
            
            # 필수 필드 검증
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        'error': f'필수 필드가 없습니다: {", ".join(missing_fields)}'
                    }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def rate_limit(calls_per_minute=60):
    """Rate limiting 데코레이터"""
    request_times = {}
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = session.get('user_id', request.remote_addr)
            current_time = time.time()
            
            # 사용자별 요청 시간 기록
            if user_id not in request_times:
                request_times[user_id] = []
            
            # 1분 이전 요청 제거
            request_times[user_id] = [
                t for t in request_times[user_id] 
                if current_time - t < 60
            ]
            
            # Rate limit 확인
            if len(request_times[user_id]) >= calls_per_minute:
                return jsonify({
                    'error': '요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.'
                }), 429
            
            # 현재 요청 시간 추가
            request_times[user_id].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def cache_response(timeout=300):
    """응답 캐싱 데코레이터"""
    cache = {}
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 캐시 키 생성
            cache_key = f"{request.path}:{request.query_string.decode()}"
            
            # 캐시 확인
            if cache_key in cache:
                cached_data, cached_time = cache[cache_key]
                if time.time() - cached_time < timeout:
                    return cached_data
            
            # 함수 실행
            result = f(*args, **kwargs)
            
            # 성공 응답만 캐싱
            if isinstance(result, tuple) and result[1] == 200:
                cache[cache_key] = (result, time.time())
            elif hasattr(result, 'status_code') and result.status_code == 200:
                cache[cache_key] = (result, time.time())
            
            return result
        return decorated_function
    return decorator