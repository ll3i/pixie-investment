"""
학습 컨텐츠 Blueprint
"""

from flask import Blueprint

# Blueprint 생성
learning_bp = Blueprint('learning', __name__, url_prefix='/api/learning')

# 라우트 임포트
from . import routes