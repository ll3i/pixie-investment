"""
알림 관련 서비스
투자 위험 알림, 시장 변동 알림 등 관리
"""

import json
from datetime import datetime, timedelta
from services.database_service import DatabaseService

class AlertService:
    def __init__(self):
        self.db = DatabaseService()
        
        # 알림 유형 정의
        self.alert_types = {
            'price_change': '가격 변동',
            'risk_warning': '투자 위험',
            'news_alert': '주요 뉴스',
            'portfolio_suggestion': '포트폴리오 제안',
            'market_analysis': '시장 분석',
            'dividend': '배당 정보',
            'earnings': '실적 발표'
        }
        
        # 알림 레벨 정의
        self.alert_levels = {
            'info': {'name': '정보', 'color': 'blue', 'priority': 1},
            'warning': {'name': '주의', 'color': 'yellow', 'priority': 2},
            'critical': {'name': '위험', 'color': 'red', 'priority': 3}
        }
    
    def get_user_alerts(self, user_id, level=None, days=7, limit=50, unread_only=False):
        """사용자 알림 목록 조회"""
        try:
            # 날짜 계산
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # 쿼리 생성
            query = '''
                SELECT id, alert_type, message, level, is_read, created_at
                FROM alerts_history
                WHERE user_id = ? AND created_at >= ?
            '''
            params = [user_id, start_date]
            
            # 조건 추가
            if level:
                query += ' AND level = ?'
                params.append(level)
            
            if unread_only:
                query += ' AND is_read = 0'
            
            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)
            
            rows = self.db.fetch_all(query, params)
            
            alerts = []
            for row in rows:
                alerts.append({
                    'id': row[0],
                    'type': row[1],
                    'type_name': self.alert_types.get(row[1], row[1]),
                    'message': row[2],
                    'level': row[3],
                    'level_info': self.alert_levels.get(row[3], {}),
                    'is_read': bool(row[4]),
                    'created_at': row[5]
                })
            
            return alerts
            
        except Exception as e:
            print(f"알림 조회 오류: {e}")
            return []
    
    def get_alert_history(self, user_id, start_date=None, end_date=None, 
                         alert_type=None, limit=100, offset=0):
        """알림 이력 조회"""
        try:
            query = 'SELECT * FROM alerts_history WHERE user_id = ?'
            params = [user_id]
            
            if start_date:
                query += ' AND created_at >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND created_at <= ?'
                params.append(end_date)
            
            if alert_type:
                query += ' AND alert_type = ?'
                params.append(alert_type)
            
            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            rows = self.db.fetch_all(query, params)
            
            history = []
            for row in rows:
                history.append({
                    'id': row[0],
                    'user_id': row[1],
                    'type': row[2],
                    'type_name': self.alert_types.get(row[2], row[2]),
                    'message': row[3],
                    'level': row[4],
                    'level_info': self.alert_levels.get(row[4], {}),
                    'is_read': bool(row[5]),
                    'created_at': row[6]
                })
            
            return history
            
        except Exception as e:
            print(f"알림 이력 조회 오류: {e}")
            return []
    
    def get_unread_count(self, user_id):
        """읽지 않은 알림 개수"""
        try:
            result = self.db.fetch_one(
                'SELECT COUNT(*) FROM alerts_history WHERE user_id = ? AND is_read = 0',
                (user_id,)
            )
            return result[0] if result else 0
            
        except Exception as e:
            print(f"미읽음 개수 조회 오류: {e}")
            return 0
    
    def mark_read(self, user_id, alert_ids):
        """알림을 읽음으로 표시"""
        try:
            updated_count = 0
            for alert_id in alert_ids:
                self.db.execute_query(
                    'UPDATE alerts_history SET is_read = 1 WHERE user_id = ? AND id = ?',
                    (user_id, alert_id)
                )
                updated_count += 1
            
            return updated_count
            
        except Exception as e:
            print(f"읽음 처리 오류: {e}")
            return 0
    
    def mark_all_read(self, user_id):
        """모든 알림을 읽음으로 표시"""
        try:
            self.db.execute_query(
                'UPDATE alerts_history SET is_read = 1 WHERE user_id = ? AND is_read = 0',
                (user_id,)
            )
            
            # 업데이트된 행 수 반환
            result = self.db.fetch_one(
                'SELECT changes()'
            )
            return result[0] if result else 0
            
        except Exception as e:
            print(f"전체 읽음 처리 오류: {e}")
            return 0
    
    def create_alert(self, user_id, alert_type, message, level, metadata=None):
        """새 알림 생성"""
        try:
            # alerts_history 테이블에 저장
            alert_id = self.db.execute_query('''
                INSERT INTO alerts_history (user_id, alert_type, message, level, is_read)
                VALUES (?, ?, ?, ?, 0)
            ''', (user_id, alert_type, message, level))
            
            # metadata가 있으면 별도 처리 (실제로는 별도 테이블이나 JSON 컬럼)
            
            return alert_id
            
        except Exception as e:
            print(f"알림 생성 오류: {e}")
            return None
    
    def delete_alert(self, user_id, alert_id):
        """알림 삭제"""
        try:
            self.db.execute_query(
                'DELETE FROM alerts_history WHERE user_id = ? AND id = ?',
                (user_id, alert_id)
            )
            
            result = self.db.fetch_one('SELECT changes()')
            return result[0] > 0 if result else False
            
        except Exception as e:
            print(f"알림 삭제 오류: {e}")
            return False
    
    def get_user_settings(self, user_id):
        """사용자 알림 설정 조회"""
        # 실제로는 별도의 설정 테이블에서 조회
        # 여기서는 기본값 반환
        return {
            'enabled': True,
            'email_notifications': False,
            'push_notifications': True,
            'alert_types': {
                'price_change': True,
                'risk_warning': True,
                'news_alert': True,
                'portfolio_suggestion': True,
                'market_analysis': False,
                'dividend': True,
                'earnings': True
            },
            'quiet_hours': {
                'enabled': False,
                'start': '22:00',
                'end': '08:00'
            }
        }
    
    def update_user_settings(self, user_id, settings):
        """사용자 알림 설정 업데이트"""
        # 실제로는 설정 테이블에 저장
        # 여기서는 받은 설정 그대로 반환
        return settings
    
    def get_alert_types(self):
        """사용 가능한 알림 유형 목록"""
        return [
            {
                'id': key,
                'name': value,
                'description': self._get_type_description(key)
            }
            for key, value in self.alert_types.items()
        ]
    
    def subscribe_alerts(self, user_id, alert_types):
        """알림 구독"""
        # 실제로는 구독 설정 저장
        return alert_types
    
    def unsubscribe_alerts(self, user_id, alert_types):
        """알림 구독 취소"""
        # 실제로는 구독 설정에서 제거
        return alert_types
    
    def get_alert_count(self, user_id, alert_type=None):
        """알림 총 개수"""
        try:
            query = 'SELECT COUNT(*) FROM alerts_history WHERE user_id = ?'
            params = [user_id]
            
            if alert_type:
                query += ' AND alert_type = ?'
                params.append(alert_type)
            
            result = self.db.fetch_one(query, params)
            return result[0] if result else 0
            
        except Exception as e:
            print(f"알림 개수 조회 오류: {e}")
            return 0
    
    def create_system_alerts(self, alert_type, message, level, target_users=None):
        """시스템 알림 생성 (전체 또는 특정 사용자들)"""
        try:
            if target_users:
                # 특정 사용자들에게만
                for user_id in target_users:
                    self.create_alert(user_id, alert_type, message, level)
            else:
                # 전체 사용자에게 (실제로는 활성 사용자 조회 필요)
                # 여기서는 예시로 구현
                pass
            
            return True
            
        except Exception as e:
            print(f"시스템 알림 생성 오류: {e}")
            return False
    
    def _get_type_description(self, alert_type):
        """알림 유형별 설명"""
        descriptions = {
            'price_change': '설정한 가격 조건에 도달했을 때 알림',
            'risk_warning': '포트폴리오 위험도가 높아졌을 때 알림',
            'news_alert': '관심 종목 관련 주요 뉴스 알림',
            'portfolio_suggestion': 'AI가 추천하는 포트폴리오 변경 제안',
            'market_analysis': '시장 전체 분석 및 전망 알림',
            'dividend': '보유 종목 배당 관련 알림',
            'earnings': '관심 종목 실적 발표 알림'
        }
        return descriptions.get(alert_type, '기타 알림')