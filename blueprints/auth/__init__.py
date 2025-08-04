"""
인증 및 사용자 관리 Blueprint
"""

from flask import Blueprint

# Blueprint 생성
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

# 라우트 임포트
from . import routes