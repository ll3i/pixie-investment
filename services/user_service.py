"""
사용자 관련 서비스
사용자 프로필, 설문조사 결과 등 관리
"""

import sqlite3
import json
from datetime import datetime
import os
import sys

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.db_client import get_supabase_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

class UserService:
    def __init__(self, db_path='newsbot.db'):
        self.db_path = db_path
        self.supabase = get_supabase_client() if SUPABASE_AVAILABLE else None
    
    def _get_connection(self):
        """데이터베이스 연결 반환"""
        return sqlite3.connect(self.db_path)
    
    def has_completed_survey(self, user_id):
        """사용자가 설문조사를 완료했는지 확인"""
        try:
            # Supabase 확인
            if self.supabase:
                try:
                    res = self.supabase.table('user_profiles').select('profile_json').eq(
                        'user_id', user_id
                    ).order('created_at', desc=True).limit(1).execute()
                    return bool(res.data and len(res.data) > 0 and res.data[0].get('profile_json'))
                except:
                    pass
            
            # SQLite 확인
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''
                SELECT COUNT(*) FROM user_profiles 
                WHERE user_id = ? AND analysis_result IS NOT NULL
            ''', (user_id,))
            count = c.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            print(f"설문 완료 확인 오류: {e}")
            return False
    
    def has_profile(self, user_id):
        """사용자 프로필 존재 여부 확인"""
        return self.has_completed_survey(user_id)
    
    def get_profile_summary(self, user_id):
        """사용자 프로필 요약 정보 반환"""
        try:
            # Supabase에서 가져오기
            if self.supabase:
                try:
                    res = self.supabase.table('user_profiles').select('*').eq(
                        'user_id', user_id
                    ).order('created_at', desc=True).limit(1).execute()
                    
                    if res.data and len(res.data) > 0:
                        profile = res.data[0]
                        return {
                            'risk_tolerance': profile.get('risk_tolerance'),
                            'investment_time_horizon': profile.get('investment_time_horizon'),
                            'financial_goal_orientation': profile.get('financial_goal_orientation'),
                            'information_processing_style': profile.get('information_processing_style'),
                            'summary': profile.get('summary', '')
                        }
                except:
                    pass
            
            # SQLite에서 가져오기
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''
                SELECT risk_tolerance, investment_time_horizon, 
                       financial_goal_orientation, information_processing_style,
                       analysis_result
                FROM user_profiles 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (user_id,))
            
            row = c.fetchone()
            conn.close()
            
            if row:
                analysis = json.loads(row[4]) if row[4] else {}
                return {
                    'risk_tolerance': row[0],
                    'investment_time_horizon': row[1],
                    'financial_goal_orientation': row[2],
                    'information_processing_style': row[3],
                    'summary': analysis.get('overall_analysis', '')
                }
            
            return None
            
        except Exception as e:
            print(f"프로필 요약 조회 오류: {e}")
            return None
    
    def get_full_profile(self, user_id):
        """사용자 전체 프로필 정보 반환"""
        try:
            # Supabase에서 가져오기
            if self.supabase:
                try:
                    res = self.supabase.table('user_profiles').select('*').eq(
                        'user_id', user_id
                    ).order('created_at', desc=True).limit(1).execute()
                    
                    if res.data and len(res.data) > 0:
                        profile = res.data[0]
                        return profile.get('profile_json', {})
                except:
                    pass
            
            # SQLite에서 가져오기
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('''
                SELECT analysis_result
                FROM user_profiles 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (user_id,))
            
            row = c.fetchone()
            conn.close()
            
            if row and row[0]:
                return json.loads(row[0])
            
            return None
            
        except Exception as e:
            print(f"전체 프로필 조회 오류: {e}")
            return None
    
    def get_latest_survey_result(self, user_id):
        """최신 설문조사 결과 반환"""
        return self.get_full_profile(user_id)
    
    def create_or_update_user(self, user_id, email=None, name=None):
        """사용자 생성 또는 업데이트"""
        try:
            conn = self._get_connection()
            c = conn.cursor()
            
            # 사용자 존재 확인
            c.execute('SELECT id FROM users WHERE id = ?', (user_id,))
            exists = c.fetchone()
            
            if exists:
                # 업데이트
                if email or name:
                    updates = []
                    params = []
                    
                    if email:
                        updates.append('email = ?')
                        params.append(email)
                    if name:
                        updates.append('name = ?')
                        params.append(name)
                    
                    updates.append('last_login = CURRENT_TIMESTAMP')
                    params.append(user_id)
                    
                    c.execute(f'''
                        UPDATE users 
                        SET {', '.join(updates)}
                        WHERE id = ?
                    ''', params)
            else:
                # 생성
                c.execute('''
                    INSERT INTO users (id, email, name, created_at, last_login)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''', (user_id, email, name))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"사용자 생성/업데이트 오류: {e}")
            return False