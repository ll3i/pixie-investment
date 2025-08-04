"""
알림 관리 Blueprint
"""

from flask import Blueprint

# Blueprint 생성
alerts_bp = Blueprint('alerts', __name__, url_prefix='/api/alerts')

# 라우트 임포트
from . import routes