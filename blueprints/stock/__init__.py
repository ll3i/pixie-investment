"""
주식 및 포트폴리오 관리 Blueprint
"""

from flask import Blueprint

# Blueprint 생성
stock_bp = Blueprint('stock', __name__, url_prefix='/api/stock')

# 라우트 임포트
from . import routes