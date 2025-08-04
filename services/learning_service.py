"""
학습 관련 서비스
투자 교육 콘텐츠 관리 및 진행 상황 추적
"""

import json
import random
from datetime import datetime, timedelta
from services.database_service import DatabaseService

class LearningService:
    def __init__(self):
        self.db = DatabaseService()
        self.terms_data = self._load_terms_data()
        self.quiz_data = self._load_quiz_data()
    
    def _load_terms_data(self):
        """투자 용어 데이터 로드"""
        # 실제로는 데이터베이스나 파일에서 로드
        return [
            {
                'id': 1,
                'term': 'PER',
                'full_name': 'Price Earnings Ratio',
                'korean': '주가수익비율',
                'description': '주가를 주당순이익으로 나눈 값으로, 주가가 기업 이익의 몇 배인지를 나타냅니다.',
                'category': 'valuation',
                'difficulty': 'beginner',
                'example': 'PER이 10이면 투자금 회수에 10년이 걸린다는 의미입니다.'
            },
            {
                'id': 2,
                'term': 'PBR',
                'full_name': 'Price Book-value Ratio',
                'korean': '주가순자산비율',
                'description': '주가를 주당순자산가치로 나눈 값으로, 기업의 자산가치 대비 주가 수준을 나타냅니다.',
                'category': 'valuation',
                'difficulty': 'beginner',
                'example': 'PBR이 1보다 낮으면 주가가 장부가치보다 낮게 거래되고 있음을 의미합니다.'
            },
            {
                'id': 3,
                'term': 'ROE',
                'full_name': 'Return On Equity',
                'korean': '자기자본이익률',
                'description': '당기순이익을 자기자본으로 나눈 값으로, 투자한 자본 대비 수익성을 나타냅니다.',
                'category': 'profitability',
                'difficulty': 'intermediate',
                'example': 'ROE가 15%면 투자한 자본 100원당 15원의 이익을 창출한다는 의미입니다.'
            }
        ]
    
    def _load_quiz_data(self):
        """퀴즈 데이터 로드"""
        return [
            {
                'id': 1,
                'question': 'PER이 낮을수록 좋은 이유는?',
                'options': [
                    '주가가 이익 대비 저평가되어 있을 가능성',
                    '기업의 성장성이 높음',
                    '배당금이 많음',
                    '거래량이 많음'
                ],
                'correct': 0,
                'explanation': 'PER이 낮다는 것은 주가가 기업 이익에 비해 상대적으로 저평가되어 있을 가능성을 의미합니다.',
                'category': 'valuation',
                'difficulty': 'beginner'
            },
            {
                'id': 2,
                'question': '다음 중 기업의 수익성을 나타내는 지표는?',
                'options': ['PER', 'PBR', 'ROE', 'EPS'],
                'correct': 2,
                'explanation': 'ROE(자기자본이익률)는 투자한 자본 대비 얼마나 많은 이익을 창출하는지를 나타내는 수익성 지표입니다.',
                'category': 'profitability',
                'difficulty': 'beginner'
            }
        ]
    
    def get_terms(self, category='all', search=''):
        """투자 용어 목록 조회"""
        terms = self.terms_data
        
        # 카테고리 필터링
        if category != 'all':
            terms = [t for t in terms if t['category'] == category]
        
        # 검색어 필터링
        if search:
            search_lower = search.lower()
            terms = [
                t for t in terms 
                if search_lower in t['term'].lower() or 
                   search_lower in t['korean'].lower() or
                   search_lower in t['description'].lower()
            ]
        
        return terms
    
    def get_term_detail(self, term_id):
        """특정 용어 상세 정보"""
        for term in self.terms_data:
            if term['id'] == term_id:
                return term
        return None
    
    def get_quiz_questions(self, category='all', difficulty='all', count=5):
        """퀴즈 문제 가져오기"""
        questions = self.quiz_data
        
        # 필터링
        if category != 'all':
            questions = [q for q in questions if q['category'] == category]
        if difficulty != 'all':
            questions = [q for q in questions if q['difficulty'] == difficulty]
        
        # 랜덤 선택
        selected = random.sample(questions, min(count, len(questions)))
        
        # 퀴즈 ID 생성
        quiz_id = f"quiz_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            'quiz_id': quiz_id,
            'questions': selected,
            'total': len(selected),
            'time_limit': len(selected) * 60  # 문제당 60초
        }
    
    def grade_quiz(self, user_id, quiz_id, answers):
        """퀴즈 채점"""
        # 실제로는 퀴즈 ID로 문제를 조회해서 채점
        # 여기서는 간단히 구현
        correct_count = 0
        total = len(answers)
        
        for i, answer in enumerate(answers):
            # 실제로는 퀴즈 ID로 정답 확인
            if i < len(self.quiz_data) and answer == self.quiz_data[i]['correct']:
                correct_count += 1
        
        score = int((correct_count / total) * 100) if total > 0 else 0
        
        # 진행 상황 저장
        self._save_quiz_result(user_id, quiz_id, score)
        
        return {
            'score': score,
            'correct': correct_count,
            'total': total,
            'passed': score >= 70,
            'message': '축하합니다! 합격입니다.' if score >= 70 else '다시 도전해보세요!'
        }
    
    def get_user_progress(self, user_id):
        """사용자 학습 진행 상황"""
        try:
            # 학습 진행 상황 조회
            query = '''
                SELECT module_id, progress, completed, score, last_accessed
                FROM learning_progress
                WHERE user_id = ?
                ORDER BY last_accessed DESC
            '''
            
            rows = self.db.fetch_all(query, (user_id,))
            
            # 전체 진행률 계산
            total_modules = 10  # 전체 모듈 수
            completed_modules = sum(1 for row in rows if row[2])  # completed가 True인 것
            
            progress_data = {
                'overall_progress': int((completed_modules / total_modules) * 100),
                'completed_modules': completed_modules,
                'total_modules': total_modules,
                'modules': [
                    {
                        'module_id': row[0],
                        'progress': row[1],
                        'completed': bool(row[2]),
                        'score': row[3],
                        'last_accessed': row[4]
                    }
                    for row in rows
                ],
                'achievements': self._get_achievements(user_id, completed_modules)
            }
            
            return progress_data
            
        except Exception as e:
            print(f"진행 상황 조회 오류: {e}")
            return {
                'overall_progress': 0,
                'completed_modules': 0,
                'total_modules': 10,
                'modules': []
            }
    
    def mark_term_completed(self, user_id, term_id):
        """용어 학습 완료 처리"""
        try:
            module_id = f"term_{term_id}"
            
            # 기존 진행 상황 확인
            existing = self.db.fetch_one(
                'SELECT id FROM learning_progress WHERE user_id = ? AND module_id = ?',
                (user_id, module_id)
            )
            
            if existing:
                # 업데이트
                self.db.execute_query('''
                    UPDATE learning_progress 
                    SET progress = 100, completed = 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND module_id = ?
                ''', (user_id, module_id))
            else:
                # 새로 생성
                self.db.execute_query('''
                    INSERT INTO learning_progress 
                    (user_id, module_id, progress, completed)
                    VALUES (?, ?, 100, 1)
                ''', (user_id, module_id))
            
            return self.get_user_progress(user_id)
            
        except Exception as e:
            print(f"학습 완료 처리 오류: {e}")
            return None
    
    def update_cardnews_progress(self, user_id, cardnews_id, current_page, total_pages):
        """카드뉴스 진행 상황 업데이트"""
        progress = int((current_page / total_pages) * 100)
        module_id = f"cardnews_{cardnews_id}"
        
        try:
            # 진행 상황 업데이트
            existing = self.db.fetch_one(
                'SELECT id FROM learning_progress WHERE user_id = ? AND module_id = ?',
                (user_id, module_id)
            )
            
            if existing:
                self.db.execute_query('''
                    UPDATE learning_progress 
                    SET progress = ?, last_accessed = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND module_id = ?
                ''', (progress, user_id, module_id))
            else:
                self.db.execute_query('''
                    INSERT INTO learning_progress 
                    (user_id, module_id, progress, completed)
                    VALUES (?, ?, ?, 0)
                ''', (user_id, module_id, progress))
            
            return {
                'progress': progress,
                'current_page': current_page,
                'total_pages': total_pages
            }
            
        except Exception as e:
            print(f"카드뉴스 진행 상황 업데이트 오류: {e}")
            return None
    
    def complete_cardnews(self, user_id, cardnews_id):
        """카드뉴스 학습 완료"""
        module_id = f"cardnews_{cardnews_id}"
        return self.mark_term_completed(user_id, module_id)
    
    def get_recommendations(self, user_id):
        """맞춤형 학습 콘텐츠 추천"""
        # 사용자 진행 상황 기반 추천
        progress = self.get_user_progress(user_id)
        
        recommendations = []
        
        # 완료하지 않은 기초 용어 추천
        if progress['overall_progress'] < 30:
            recommendations.append({
                'type': 'terms',
                'title': '필수 투자 용어 학습',
                'description': '투자를 시작하기 전 꼭 알아야 할 기본 용어들',
                'difficulty': 'beginner',
                'estimated_time': '15분'
            })
        
        # 퀴즈 추천
        recommendations.append({
            'type': 'quiz',
            'title': '투자 기초 퀴즈',
            'description': '학습한 내용을 확인해보세요',
            'difficulty': 'beginner',
            'estimated_time': '10분'
        })
        
        return {
            'recommendations': recommendations,
            'reason': '현재 학습 진행도를 기반으로 추천합니다.'
        }
    
    def get_user_achievements(self, user_id):
        """사용자 성취도 및 뱃지"""
        progress = self.get_user_progress(user_id)
        
        return {
            'level': self._calculate_level(progress['completed_modules']),
            'badges': self._get_achievements(user_id, progress['completed_modules']),
            'points': progress['completed_modules'] * 100,
            'next_level_progress': (progress['completed_modules'] % 3) * 33
        }
    
    def get_leaderboard(self, period='week', limit=10):
        """학습 리더보드"""
        # 실제로는 기간별 쿼리
        # 여기서는 더미 데이터
        return [
            {
                'rank': i + 1,
                'user_id': f'user_{i}',
                'nickname': f'투자고수{i}',
                'points': 1000 - (i * 50),
                'level': 5 - (i // 2)
            }
            for i in range(limit)
        ]
    
    def _save_quiz_result(self, user_id, quiz_id, score):
        """퀴즈 결과 저장"""
        try:
            self.db.execute_query('''
                INSERT INTO learning_progress 
                (user_id, module_id, progress, completed, score)
                VALUES (?, ?, 100, ?, ?)
            ''', (user_id, quiz_id, score >= 70, score))
        except Exception as e:
            print(f"퀴즈 결과 저장 오류: {e}")
    
    def _calculate_level(self, completed_modules):
        """레벨 계산"""
        if completed_modules >= 30:
            return 5
        elif completed_modules >= 20:
            return 4
        elif completed_modules >= 10:
            return 3
        elif completed_modules >= 5:
            return 2
        else:
            return 1
    
    def _get_achievements(self, user_id, completed_modules):
        """성취 뱃지 목록"""
        badges = []
        
        if completed_modules >= 1:
            badges.append({
                'id': 'first_step',
                'name': '첫 걸음',
                'description': '첫 학습을 완료했습니다!',
                'icon': '🎯'
            })
        
        if completed_modules >= 5:
            badges.append({
                'id': 'eager_learner',
                'name': '열정적인 학습자',
                'description': '5개 모듈을 완료했습니다!',
                'icon': '🔥'
            })
        
        if completed_modules >= 10:
            badges.append({
                'id': 'investment_expert',
                'name': '투자 전문가',
                'description': '10개 모듈을 완료했습니다!',
                'icon': '💎'
            })
        
        return badges