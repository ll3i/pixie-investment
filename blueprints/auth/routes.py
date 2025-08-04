"""
인증 관련 라우트
사용자 프로필, 설문조사, 세션 관리 등
"""

from flask import render_template, request, jsonify, session, redirect, url_for
import uuid
import json
import os
from datetime import datetime
from . import auth_bp
from services.user_service import UserService
from services.survey_service import SurveyService

# 서비스 초기화
user_service = UserService()
survey_service = SurveyService()

@auth_bp.route('/survey')
def survey():
    """
    설문조사 페이지 진입 시 user_id 세션 생성 보장 및 설문 완료 여부 판별
    """
    if 'user_id' not in session:
        session['user_id'] = f"user_{str(uuid.uuid4())[:8]}"
    
    user_id = session['user_id']
    survey_completed = user_service.has_completed_survey(user_id)
    
    return render_template('survey.html', survey_completed=survey_completed)

@auth_bp.route('/submit_survey', methods=['POST'])
def submit_survey():
    """설문 조사 결과를 제출하고 결과를 분석합니다."""
    try:
        data = request.get_json()
        answers = data.get('answers', [])
        
        # user_id 세션에서 가져오기
        user_id = session.get('user_id')
        if not user_id:
            user_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = user_id
        
        # 설문 분석 및 저장
        result = survey_service.analyze_and_save_survey(user_id, answers)
        
        # 세션에 결과 저장
        session['survey_result'] = result
        session['survey_completed'] = True
        session.permanent = True
        
        return jsonify({
            'success': True,
            'redirect': url_for('auth.survey_result')
        })
        
    except Exception as e:
        print(f"설문 제출 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@auth_bp.route('/survey/result')
def survey_result():
    """설문조사 결과 페이지를 렌더링합니다."""
    # 세션 또는 DB에서 결과 가져오기
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('auth.survey'))
    
    # 세션에서 먼저 확인
    result = session.get('survey_result')
    
    # 세션에 없으면 DB에서 가져오기
    if not result:
        result = user_service.get_latest_survey_result(user_id)
    
    if not result:
        return redirect(url_for('auth.survey'))
    
    return render_template('survey_result.html', result=result)

@auth_bp.route('/reset_survey', methods=['POST'])
def reset_survey():
    """설문조사를 초기화합니다."""
    try:
        # 세션 초기화
        session.pop('survey_result', None)
        session.pop('survey_completed', None)
        
        # 새로운 user_id 생성
        session['user_id'] = f"user_{str(uuid.uuid4())[:8]}"
        
        return jsonify({
            'success': True,
            'message': '설문조사가 초기화되었습니다.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@auth_bp.route('/get_survey_result', methods=['GET'])
def get_survey_result():
    """저장된 설문조사 결과를 반환합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '사용자 ID가 없습니다.'}), 401
        
        # 세션에서 먼저 확인
        result = session.get('survey_result')
        
        # 세션에 없으면 DB에서 가져오기
        if not result:
            result = user_service.get_latest_survey_result(user_id)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': '설문조사 결과가 없습니다.'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/profile-status')
def profile_status():
    """사용자 프로필 상태를 반환합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            user_id = f"user_{str(uuid.uuid4())[:8]}"
            session['user_id'] = user_id
        
        has_profile = user_service.has_profile(user_id)
        profile_data = None
        
        if has_profile:
            profile_data = user_service.get_profile_summary(user_id)
        
        return jsonify({
            'user_id': user_id,
            'has_profile': has_profile,
            'profile': profile_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/profile')
def get_profile():
    """사용자 프로필 정보를 반환합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        profile = user_service.get_full_profile(user_id)
        if profile:
            return jsonify(profile)
        else:
            return jsonify({'error': '프로필이 없습니다.'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500