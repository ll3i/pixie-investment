"""
채팅 관련 라우트
AI 상담, 채팅 기록, 스트리밍 등
"""

from flask import request, jsonify, Response, session, stream_with_context
import json
import time
import uuid
from datetime import datetime
from . import chat_bp
from services.chat_service import ChatService
from services.user_service import UserService

# 서비스 초기화
chat_service = ChatService()
user_service = UserService()

@chat_bp.route('/chat', methods=['POST'])
def chat():
    """AI 챗봇과 대화를 처리합니다."""
    try:
        # 세션 확인
        session_id = request.headers.get('X-Session-ID') or session.get('user_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['user_id'] = session_id
        
        # 요청 데이터 파싱
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': '메시지가 없습니다.'}), 400
        
        # 사용자 프로필 확인
        user_profile = user_service.get_profile_summary(session_id)
        
        # AI 응답 생성
        def generate():
            try:
                # 상태 콜백을 포함한 스트리밍 응답
                for chunk in chat_service.generate_response_stream(
                    session_id, 
                    message, 
                    user_profile
                ):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    
            except Exception as e:
                error_msg = {
                    'type': 'error',
                    'content': f'오류가 발생했습니다: {str(e)}'
                }
                yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
            finally:
                yield "data: [DONE]\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'X-Session-ID': session_id
            }
        )
        
    except Exception as e:
        print(f"채팅 오류: {e}")
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/chat-stream')
def chat_stream():
    """SSE를 통한 실시간 채팅 스트림"""
    def generate():
        # 초기 연결 메시지
        yield f"data: {json.dumps({'type': 'connected', 'message': '연결되었습니다'}, ensure_ascii=False)}\n\n"
        
        # 주기적으로 하트비트 전송
        count = 0
        while True:
            time.sleep(30)  # 30초마다
            count += 1
            heartbeat = {
                'type': 'heartbeat',
                'timestamp': datetime.now().isoformat(),
                'count': count
            }
            yield f"data: {json.dumps(heartbeat, ensure_ascii=False)}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@chat_bp.route('/chat-history', methods=['GET', 'POST'])
def chat_history():
    """채팅 기록을 조회하거나 저장합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        if request.method == 'GET':
            # 채팅 기록 조회
            limit = request.args.get('limit', 50, type=int)
            history = chat_service.get_chat_history(user_id, limit)
            return jsonify({'history': history})
            
        else:  # POST
            # 채팅 기록 저장
            data = request.get_json()
            role = data.get('role')
            content = data.get('content')
            
            if not role or not content:
                return jsonify({'error': '필수 데이터가 없습니다.'}), 400
            
            chat_service.save_chat_message(user_id, role, content)
            return jsonify({'success': True})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/clear-chat', methods=['POST'])
def clear_chat():
    """채팅 기록을 초기화합니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        chat_service.clear_chat_history(user_id)
        return jsonify({'success': True, 'message': '채팅 기록이 초기화되었습니다.'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/export-chat', methods=['GET'])
def export_chat():
    """채팅 기록을 JSON 형식으로 내보냅니다."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
        history = chat_service.get_full_chat_history(user_id)
        
        return jsonify({
            'user_id': user_id,
            'export_date': datetime.now().isoformat(),
            'chat_history': history
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500