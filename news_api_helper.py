#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
뉴스 API 헬퍼 함수들
실시간 뉴스 수집 및 캐싱 관리
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# 전역 뉴스 캐시
news_cache = {
    'data': None,
    'last_update': None,
    'cache_duration': timedelta(minutes=30)
}

# 전역 뉴스 수집기 인스턴스
news_collector = None

async def initialize_news_collector():
    """뉴스 수집기 초기화"""
    global news_collector
    
    try:
        from src.news_collector_service import NewsCollectorService
        news_collector = NewsCollectorService()
        
        # MCP 초기화
        if await news_collector.initialize_mcp():
            logger.info("News collector initialized successfully")
            return True
        else:
            logger.warning("Failed to initialize MCP, using mock data")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize news collector: {e}")
        return False

async def collect_realtime_news() -> Optional[Dict[str, Any]]:
    """실시간 뉴스 수집"""
    global news_cache, news_collector
    
    try:
        # 수집기가 초기화되지 않았으면 초기화
        if not news_collector:
            await initialize_news_collector()
        
        # 주요 키워드로 뉴스 수집
        keywords = ["코스피", "삼성전자", "SK하이닉스", "NAVER", "카카오", "금리", "환율", "AI", "반도체", "배터리"]
        
        if news_collector:
            news_data = await news_collector.collect_daily_news(keywords)
            
            # 픽시 인사이트 생성
            insights = news_collector.generate_pixie_insights(news_data)
            news_data['pixie_insights'] = insights
            
            # 캐시 업데이트
            news_cache['data'] = news_data
            news_cache['last_update'] = datetime.now()
            
            logger.info(f"Collected {len(news_data['news_list'])} news items")
            return news_data
        else:
            # MCP를 사용할 수 없는 경우 mock 데이터 반환
            return None
            
    except Exception as e:
        logger.error(f"Failed to collect realtime news: {e}")
        return None

def get_mock_news_data() -> Dict[str, List[Dict[str, Any]]]:
    """모의 뉴스 데이터 생성"""
    current_time = datetime.now()
    
    mock_data = {
        'today_pick': [
            {
                "id": 1,
                "category": "IT",
                "title": "삼성전자 실적 발표,\n시장 예상치 상회",
                "content": "삼성전자가 발표한 최근 실적이 시장 예상치를 상회하며 긍정적인 반응을 보이고 있습니다.",
                "summary": "삼성전자가 발표한 최근 실적이 시장 예상치를 상회하며 긍정적인 반응을 보이고 있습니다.",
                "source": "한국경제",
                "time": "14시간 전",
                "likes": 93,
                "link": "#",
                "pub_date": current_time.isoformat(),
                "timestamp": current_time.isoformat(),
                "sentiment_score": 0.8,
                "sentiment": "긍정",
                "isLiked": False
            },
            {
                "id": 2,
                "category": "경제",
                "title": "'불장' 된 국장…\n해외주식 '열풍' 주춤",
                "content": "올해 상반기 해외주식 결제액이 줄고, 채권 비중이 늘며 해외투자 열기가 주춤해졌습니다.",
                "summary": "올해 상반기 해외주식 결제액이 줄고, 채권 비중이 늘며 해외투자 열기가 주춤해졌습니다.",
                "source": "경향신문",
                "time": "하루 전",
                "likes": 93,
                "link": "#",
                "pub_date": (current_time - timedelta(days=1)).isoformat(),
                "timestamp": (current_time - timedelta(days=1)).isoformat(),
                "sentiment_score": 0.3,
                "sentiment": "부정",
                "isLiked": False
            },
            {
                "id": 3,
                "category": "제약",
                "title": "코스피 제약바이오\n5월 주식거래액 6.8조 원",
                "content": "5월 코스피 제약바이오 업종 거래액이 전월 대비 약 0.96% 감소한 6조 8,428억 원을 기록했습니다.",
                "summary": "5월 코스피 제약바이오 업종 거래액이 전월 대비 약 0.96% 감소한 6조 8,428억 원을 기록했습니다.",
                "source": "의학신문",
                "time": "3일 전",
                "likes": 94,
                "link": "#",
                "pub_date": (current_time - timedelta(days=3)).isoformat(),
                "timestamp": (current_time - timedelta(days=3)).isoformat(),
                "sentiment_score": -0.2,
                "sentiment": "부정",
                "isLiked": True
            },
            {
                "id": 4,
                "category": "금융",
                "title": "KB증권, PRIME CLUB\n국내주식 투자 콘서트 개최",
                "content": "KB증권이 전국 주요 도시에서 국내주식 투자 콘서트를 개최해 투자 정보를 제공합니다.",
                "summary": "KB증권이 전국 주요 도시에서 국내주식 투자 콘서트를 개최해 투자 정보를 제공합니다.",
                "source": "여성소비자신문",
                "time": "1개월 전",
                "likes": 93,
                "link": "#",
                "pub_date": (current_time - timedelta(days=30)).isoformat(),
                "timestamp": (current_time - timedelta(days=30)).isoformat(),
                "sentiment_score": 0.6,
                "sentiment": "긍정",
                "isLiked": False
            }
        ],
        'popular': [
            {
                "id": 5,
                "category": "AI",
                "title": "국내 AI 스타트업\n대규모 투자 유치",
                "content": "국내 AI 스타트업이 글로벌 벤처캐피털로부터 1000억원 규모의 투자를 유치했습니다.",
                "summary": "AI 기술력을 인정받아 대규모 투자 유치에 성공",
                "source": "디지털타임스",
                "time": "6시간 전",
                "likes": 156,
                "sentiment_score": 0.9,
                "sentiment": "긍정"
            }
        ],
        'recommend': [
            {
                "id": 6,
                "category": "반도체",
                "title": "SK하이닉스 HBM4\n개발 순항",
                "content": "SK하이닉스가 차세대 고대역폭 메모리 HBM4 개발이 순조롭게 진행되고 있다고 발표했습니다.",
                "summary": "차세대 메모리 기술 개발로 시장 선도",
                "source": "전자신문",
                "time": "12시간 전",
                "likes": 128,
                "sentiment_score": 0.85,
                "sentiment": "긍정"
            }
        ]
    }
    
    return mock_data

def get_pixie_insights() -> Dict[str, str]:
    """픽시의 인사이트 가져오기"""
    global news_cache
    
    # 캐시된 데이터가 있으면 사용
    if news_cache['data'] and 'pixie_insights' in news_cache['data']:
        return news_cache['data']['pixie_insights']
    
    # 기본 인사이트 반환
    return {
        "insight_title": "기술과 건강의 교차점, 투자 기회를 말하다",
        "insight_content": "오늘은 의약·헬스케어 산업의 활력이 두드러졌어요. AI·바이오 기술 기반 플랫폼의 성장부터 신약 파이프라인 발표까지, "
                         "기술과 건강이 만나는 지점에서 새로운 투자 기회가 보이고 있죠. 또한 정부의 약가 정책 변화, 증권사의 투자 교육 확대 등은 "
                         "시장 참여자들에게 구조적인 영향을 미치는 제도 변화로 해석돼요.",
        "pixie_quote": "기회는 빠르게,\n정책은 조심스럽게!",
        "trend_summary": "오늘의 핵심 키워드: 반도체, AI, 바이오"
    }

def get_trend_keywords() -> List[Dict[str, Any]]:
    """트렌드 키워드 가져오기"""
    global news_cache
    
    # 캐시된 데이터가 있으면 사용
    if news_cache['data'] and 'trend_keywords' in news_cache['data']:
        keywords = news_cache['data']['trend_keywords'][:12]
        
        # UI에 맞게 분류 (filled/outlined)
        result = []
        filled_indices = [0, 3, 5, 7]  # filled로 표시할 인덱스
        
        for i, kw in enumerate(keywords):
            result.append({
                "keyword": f"#{kw['keyword']}",
                "type": "filled" if i in filled_indices else "outlined",
                "count": kw['count'],
                "score": kw.get('score', kw['count'])
            })
        
        return result
    
    # 기본 트렌드 키워드 반환
    return [
        {"keyword": "#디스플레이", "type": "filled"},
        {"keyword": "#바이오", "type": "outlined"},
        {"keyword": "#유전자 치료제", "type": "outlined"},
        {"keyword": "#테슬라 옵티머스", "type": "filled"},
        {"keyword": "#AI 반도체", "type": "outlined"},
        {"keyword": "#삼성전자", "type": "filled"},
        {"keyword": "#2차전지", "type": "outlined"},
        {"keyword": "#IRA 법안", "type": "filled"},
        {"keyword": "#탄소배출권", "type": "outlined"},
        {"keyword": "#토목 SOC", "type": "outlined"}
    ]

def get_news_sentiment_summary() -> Dict[str, Any]:
    """뉴스 감정 분석 요약"""
    global news_cache
    
    # 캐시된 데이터가 있으면 사용
    if news_cache['data'] and 'sentiment_summary' in news_cache['data']:
        return news_cache['data']['sentiment_summary']
    
    # 기본 감정 분석 요약 반환
    return {
        "total_count": 50,
        "positive_count": 20,
        "negative_count": 10,
        "neutral_count": 20,
        "average_score": 0.6,
        "overall_sentiment": "긍정적"
    }