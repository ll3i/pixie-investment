"""
채팅 및 AI 상담 Blueprint
"""

from flask import Blueprint

# Blueprint 생성
chat_bp = Blueprint('chat', __name__, url_prefix='/api')

# 라우트 임포트
from . import routes