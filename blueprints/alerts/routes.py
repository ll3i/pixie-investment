"""
알림 관련 라우트
투자 위험 알림, 시장 변동 알림, 뉴스 알림 등
"""

from flask import request, jsonify, session, render_template
from datetime import datetime, timedelta
import json
from . import alerts_bp
from services.alert_service import AlertService
from blueprints.utils.decorators import require_auth, handle_errors, log_request

# 서비스 초기화
alert_service = AlertService()

# 페이지 라우트
@alerts_bp.route('/page', endpoint='page')
def alerts_page():
    """알림 메인 페이지"""
    return render_template('alerts.html')

@alerts_bp.route('/page/history', endpoint='page_history')
def alert_history_page():
    """알림 이력 페이지"""
    return render_template('alert_history.html')

# API 라우트
@alerts_bp.route('', methods=['GET'])
@require_auth
@handle_errors
def get_alerts():
    """사용자 알림 목록 조회"""
    try:
        user_id = session.get('user_id')
        
        # 파라미터 파싱
        level = request.args.get('level')  # info, warning, critical
        days = request.args.get('days', 7, type=int)
        limit = request.args.get('limit', 50, type=int)
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        
        alerts = alert_service.get_user_alerts(
            user_id=user_id,
            level=level,
            days=days,
            limit=limit,
            unread_only=unread_only
        )
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/history', methods=['GET'])
@require_auth
def get_alert_history():
    """알림 이력 조회"""
    try:
        user_id = session.get('user_id')
        
        # 파라미터 파싱
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        alert_type = request.args.get('type')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        history = alert_service.get_alert_history(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            alert_type=alert_type,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'history': history,
            'total': alert_service.get_alert_count(user_id, alert_type)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/unread-count', methods=['GET'])
@require_auth
def get_unread_count():
    """읽지 않은 알림 개수 조회"""
    try:
        user_id = session.get('user_id')
        count = alert_service.get_unread_count(user_id)
        
        return jsonify({
            'unread_count': count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/mark-read', methods=['POST'])
@require_auth
@handle_errors
def mark_alerts_read():
    """알림을 읽음으로 표시"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        alert_ids = data.get('alert_ids', [])
        mark_all = data.get('mark_all', False)
        
        if mark_all:
            # 모든 알림을 읽음으로 처리
            result = alert_service.mark_all_read(user_id)
        elif alert_ids:
            # 특정 알림들을 읽음으로 처리
            result = alert_service.mark_read(user_id, alert_ids)
        else:
            return jsonify({'error': '알림 ID가 필요합니다.'}), 400
        
        return jsonify({
            'success': True,
            'updated_count': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/mark-read-one', methods=['POST'])
@require_auth
def mark_one_read():
    """단일 알림을 읽음으로 표시"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        alert_id = data.get('alert_id')
        
        if not alert_id:
            return jsonify({'error': '알림 ID가 필요합니다.'}), 400
        
        result = alert_service.mark_read(user_id, [alert_id])
        
        return jsonify({
            'success': True,
            'updated': result > 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/settings', methods=['GET', 'POST'])
@require_auth
def alert_settings():
    """알림 설정 조회/수정"""
    try:
        user_id = session.get('user_id')
        
        if request.method == 'GET':
            # 현재 설정 조회
            settings = alert_service.get_user_settings(user_id)
            return jsonify(settings)
            
        else:  # POST
            # 설정 업데이트
            data = request.get_json()
            updated_settings = alert_service.update_user_settings(user_id, data)
            
            return jsonify({
                'success': True,
                'settings': updated_settings
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/create', methods=['POST'])
@require_auth
@log_request
def create_alert():
    """새 알림 생성 (관리자/시스템용)"""
    try:
        data = request.get_json()
        
        # 필수 필드 검증
        required_fields = ['alert_type', 'message', 'level']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'필수 필드가 없습니다: {", ".join(missing_fields)}'
            }), 400
        
        # 알림 생성
        alert_id = alert_service.create_alert(
            user_id=data.get('user_id'),  # 특정 사용자 또는 전체
            alert_type=data['alert_type'],
            message=data['message'],
            level=data['level'],
            metadata=data.get('metadata', {})
        )
        
        return jsonify({
            'success': True,
            'alert_id': alert_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/delete/<int:alert_id>', methods=['DELETE'])
@require_auth
def delete_alert(alert_id):
    """알림 삭제"""
    try:
        user_id = session.get('user_id')
        result = alert_service.delete_alert(user_id, alert_id)
        
        if result:
            return jsonify({
                'success': True,
                'message': '알림이 삭제되었습니다.'
            })
        else:
            return jsonify({'error': '알림을 찾을 수 없습니다.'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/types', methods=['GET'])
def get_alert_types():
    """사용 가능한 알림 유형 목록"""
    try:
        alert_types = alert_service.get_alert_types()
        
        return jsonify({
            'types': alert_types
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/subscribe', methods=['POST'])
@require_auth
def subscribe_alerts():
    """특정 알림 유형 구독"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        alert_types = data.get('alert_types', [])
        
        if not alert_types:
            return jsonify({'error': '구독할 알림 유형이 필요합니다.'}), 400
        
        result = alert_service.subscribe_alerts(user_id, alert_types)
        
        return jsonify({
            'success': True,
            'subscribed': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@alerts_bp.route('/unsubscribe', methods=['POST'])
@require_auth
def unsubscribe_alerts():
    """알림 구독 취소"""
    try:
        user_id = session.get('user_id')
        data = request.get_json()
        alert_types = data.get('alert_types', [])
        
        if not alert_types:
            return jsonify({'error': '구독 취소할 알림 유형이 필요합니다.'}), 400
        
        result = alert_service.unsubscribe_alerts(user_id, alert_types)
        
        return jsonify({
            'success': True,
            'unsubscribed': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500