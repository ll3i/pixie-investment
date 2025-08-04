"""
뉴스 관리 Blueprint
"""

from flask import Blueprint

# Blueprint 생성
news_bp = Blueprint('news', __name__, url_prefix='/api/news')

# 라우트 임포트
from . import routes