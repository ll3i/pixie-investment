"""
학습 관련 라우트
투자 교육, 퀴즈, 진행 상황 추적 등
"""

from flask import request, jsonify, session, render_template
from datetime import datetime
import json
from . import learning_bp
from services.learning_service import LearningService
from blueprints.utils.decorators import require_auth, handle_errors, cache_response

# 서비스 초기화
learning_service = LearningService()

# 페이지 라우트 (Blueprint prefix 없음)
@learning_bp.route('/page', endpoint='page')
def learning_page():
    """학습 메인 페이지"""
    return render_template('learning.html')

@learning_bp.route('/page/terms', endpoint='page_terms')
def learning_terms_page():
    """투자 용어 학습 페이지"""
    return render_template('learning_terms.html')

@learning_bp.route('/page/term/<term_name>', endpoint='page_term')
def learning_term_page(term_name):
    """개별 용어 학습 페이지"""
    return render_template('learning_term.html', term_name=term_name)

@learning_bp.route('/page/quiz', endpoint='page_quiz')
def learning_quiz_page():
    """투자 퀴즈 페이지"""
    return render_template('learning_quiz.html')

@learning_bp.route('/page/<module>', endpoint='page_module')
def learning_module_page(module):
    """학습 모듈별 페이지"""
    module_templates = {
        'terms': 'learning_terms.html',
        'quiz': 'learning_quiz.html',
        'cardnews': 'learning_cardnews.html'
    }
    
    template = module_templates.get(module)
    if template:
        return render_template(template)
    else:
        return render_template('learning.html')

# API 라우트
@learning_bp.route('/terms')
@cache_response(timeout=3600)  # 1시간 캐시
def get_learning_terms():
    """투자 용어 목록 조회"""
    try:
        category = request.args.get('category', 'all')
        search = request.args.get('search', '')
        
        terms = learning_service.get_terms(category, search)
        
        return jsonify({
            'terms': terms,
            'count': len(terms)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/term/<int:term_id>')
@cache_response(timeout=3600)
def get_learning_term(term_id):
    """특정 투자 용어 상세 정보"""
    try:
        term = learning_service.get_term_detail(term_id)
        
        if not term:
            return jsonify({'error': '용어를 찾을 수 없습니다.'}), 404
        
        return jsonify(term)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/quiz')
@handle_errors
def get_quiz():
    """투자 퀴즈 문제 조회"""
    try:
        category = request.args.get('category', 'all')
        difficulty = request.args.get('difficulty', 'medium')
        count = request.args.get('count', 5, type=int)
        
        quiz_data = learning_service.get_quiz_questions(
            category=category,
            difficulty=difficulty,
            count=count
        )
        
        return jsonify(quiz_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/quiz/submit', methods=['POST'])
@require_auth
@handle_errors
def submit_quiz():
    """퀴즈 답안 제출 및 채점"""
    try:
        data = request.get_json()
        quiz_id = data.get('quiz_id')
        answers = data.get('answers', [])
        
        if not quiz_id or not answers:
            return jsonify({'error': '필수 데이터가 없습니다.'}), 400
        
        user_id = session.get('user_id')
        result = learning_service.grade_quiz(user_id, quiz_id, answers)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/progress')
@require_auth
def get_learning_progress():
    """학습 진행 상황 조회"""
    try:
        user_id = session.get('user_id')
        progress = learning_service.get_user_progress(user_id)
        
        return jsonify(progress)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/complete-term', methods=['POST'])
@require_auth
def complete_term():
    """용어 학습 완료 처리"""
    try:
        data = request.get_json()
        term_id = data.get('term_id')
        
        if not term_id:
            return jsonify({'error': '용어 ID가 필요합니다.'}), 400
        
        user_id = session.get('user_id')
        result = learning_service.mark_term_completed(user_id, term_id)
        
        return jsonify({
            'success': True,
            'message': '학습이 완료되었습니다.',
            'progress': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/cardnews/progress', methods=['POST'])
@require_auth
def update_cardnews_progress():
    """카드뉴스 진행 상황 업데이트"""
    try:
        data = request.get_json()
        cardnews_id = data.get('cardnews_id')
        current_page = data.get('current_page')
        total_pages = data.get('total_pages')
        
        if not all([cardnews_id, current_page is not None, total_pages]):
            return jsonify({'error': '필수 데이터가 없습니다.'}), 400
        
        user_id = session.get('user_id')
        result = learning_service.update_cardnews_progress(
            user_id, cardnews_id, current_page, total_pages
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/cardnews/complete', methods=['POST'])
@require_auth
def complete_cardnews():
    """카드뉴스 학습 완료"""
    try:
        data = request.get_json()
        cardnews_id = data.get('cardnews_id')
        
        if not cardnews_id:
            return jsonify({'error': '카드뉴스 ID가 필요합니다.'}), 400
        
        user_id = session.get('user_id')
        result = learning_service.complete_cardnews(user_id, cardnews_id)
        
        return jsonify({
            'success': True,
            'message': '카드뉴스 학습이 완료되었습니다.',
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/recommend')
@require_auth
def get_learning_recommendations():
    """맞춤형 학습 콘텐츠 추천"""
    try:
        user_id = session.get('user_id')
        recommendations = learning_service.get_recommendations(user_id)
        
        return jsonify(recommendations)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/achievements')
@require_auth
def get_achievements():
    """학습 성취도 및 뱃지 조회"""
    try:
        user_id = session.get('user_id')
        achievements = learning_service.get_user_achievements(user_id)
        
        return jsonify(achievements)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@learning_bp.route('/leaderboard')
@cache_response(timeout=300)  # 5분 캐시
def get_leaderboard():
    """학습 리더보드 조회"""
    try:
        period = request.args.get('period', 'week')  # week, month, all
        limit = request.args.get('limit', 10, type=int)
        
        leaderboard = learning_service.get_leaderboard(period, limit)
        
        return jsonify({
            'period': period,
            'leaderboard': leaderboard
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500