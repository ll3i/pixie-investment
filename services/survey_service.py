"""
설문조사 분석 서비스
AI를 사용한 투자 성향 분석
"""

import json
import os
import sys
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.investment_advisor import InvestmentAdvisor
    from src.user_profile_analyzer import analyze_survey_responses_with_ai
    from src.db_client import get_supabase_client
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    print("분석 모듈을 사용할 수 없습니다.")

class SurveyService:
    def __init__(self):
        self.analysis_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "analysis_results.json"
        )
        self.supabase = get_supabase_client() if ANALYSIS_AVAILABLE else None
    
    def analyze_and_save_survey(self, user_id, answers):
        """설문조사 응답을 분석하고 저장"""
        try:
            if not ANALYSIS_AVAILABLE:
                # 모듈이 없을 경우 기본 분석
                return self._basic_analysis(answers)
            
            # AI를 사용한 분석
            scores = analyze_survey_responses_with_ai(answers)
            
            # 상세 분석 생성
            detailed_analysis = self._generate_detailed_analysis(scores, answers)
            overall_analysis = self._generate_overall_analysis(scores)
            
            # 포트폴리오 추천
            portfolio_result = self._recommend_portfolio(scores)
            
            result = {
                'scores': scores,
                'overall_analysis': overall_analysis,
                'detailed_analysis': detailed_analysis,
                'portfolio': portfolio_result['portfolio'],
                'portfolio_reason': portfolio_result['reason']
            }
            
            # 파일에 저장
            self._save_to_file(result)
            
            # Supabase에 저장
            if self.supabase:
                try:
                    self.supabase.table("user_profiles").upsert([{
                        "user_id": user_id,
                        "profile_json": result,
                        "summary": overall_analysis,
                        "risk_tolerance": scores.get('risk_tolerance'),
                        "investment_time_horizon": scores.get('investment_time_horizon'),
                        "financial_goal_orientation": scores.get('financial_goal_orientation'),
                        "information_processing_style": scores.get('information_processing_style')
                    }]).execute()
                except Exception as e:
                    print(f"Supabase 저장 오류: {e}")
            
            return result
            
        except Exception as e:
            print(f"설문 분석 오류: {e}")
            return self._basic_analysis(answers)
    
    def _basic_analysis(self, answers):
        """기본 분석 (AI 사용 불가 시)"""
        # 간단한 점수 계산
        scores = {
            'risk_tolerance': 0,
            'investment_time_horizon': 0,
            'financial_goal_orientation': 0,
            'information_processing_style': 0
        }
        
        # 답변 기반 기본 점수 계산
        for i, answer in enumerate(answers):
            if i < 3:  # 위험 성향
                scores['risk_tolerance'] += answer - 3
            elif i < 5:  # 투자 기간
                scores['investment_time_horizon'] += answer - 3
            elif i < 7:  # 목표 지향
                scores['financial_goal_orientation'] += answer - 3
            else:  # 정보 처리
                scores['information_processing_style'] += answer - 3
        
        # 정규화
        for key in scores:
            scores[key] = max(-2, min(2, scores[key] / 2))
        
        return {
            'scores': scores,
            'overall_analysis': '기본 분석 결과입니다.',
            'detailed_analysis': '상세 분석은 AI 모듈이 필요합니다.',
            'portfolio': [{
                'name': '균형형 포트폴리오',
                'description': '다양한 자산에 분산 투자하는 포트폴리오',
                'assets': [
                    {'name': '국내 주식', 'allocation': 30},
                    {'name': '해외 주식', 'allocation': 20},
                    {'name': '채권', 'allocation': 30},
                    {'name': '현금', 'allocation': 20}
                ]
            }],
            'portfolio_reason': '균형적인 투자 성향'
        }
    
    def _generate_detailed_analysis(self, scores, answers):
        """상세 분석 생성"""
        analysis = []
        
        # 위험 성향 분석
        risk = scores.get('risk_tolerance', 0)
        if risk >= 1:
            analysis.append("높은 위험을 감수할 수 있는 공격적 투자자입니다.")
        elif risk >= -0.5:
            analysis.append("중간 수준의 위험을 선호하는 균형형 투자자입니다.")
        else:
            analysis.append("안정성을 중시하는 보수적 투자자입니다.")
        
        # 투자 기간 분석
        horizon = scores.get('investment_time_horizon', 0)
        if horizon >= 1:
            analysis.append("장기 투자를 선호하여 복리 효과를 극대화할 수 있습니다.")
        elif horizon >= -0.5:
            analysis.append("중기 투자를 통해 안정적 수익을 추구합니다.")
        else:
            analysis.append("단기 투자로 빠른 수익 실현을 선호합니다.")
        
        return " ".join(analysis)
    
    def _generate_overall_analysis(self, scores):
        """종합 분석 생성"""
        risk_level = "고위험" if scores.get('risk_tolerance', 0) >= 1 else "중위험" if scores.get('risk_tolerance', 0) >= -0.5 else "저위험"
        time_horizon = "장기" if scores.get('investment_time_horizon', 0) >= 1 else "중기" if scores.get('investment_time_horizon', 0) >= -0.5 else "단기"
        
        return f"귀하는 {risk_level} {time_horizon} 투자자로 분석되었습니다."
    
    def _recommend_portfolio(self, scores):
        """포트폴리오 추천"""
        risk = scores.get('risk_tolerance', 0)
        horizon = scores.get('investment_time_horizon', 0)
        goal = scores.get('financial_goal_orientation', 0)
        process = scores.get('information_processing_style', 0)
        
        # 추천 근거 생성
        reason = []
        if risk >= 1:
            reason.append('고위험 선호')
        elif risk >= -0.5:
            reason.append('중간 위험 선호')
        else:
            reason.append('저위험 선호')
        
        if horizon >= 1:
            reason.append('장기 투자 지향')
        elif horizon >= -0.5:
            reason.append('중기 투자 지향')
        else:
            reason.append('단기 투자 지향')
        
        reason_text = ' · '.join(reason)
        
        # 포트폴리오 결정
        if risk >= 1 and horizon >= 0.5:
            portfolio = [{
                'name': '공격적 성장 포트폴리오',
                'description': '고위험·장기 투자 성향에 맞춘 성장주 중심 포트폴리오',
                'assets': [
                    {'name': '국내 성장주', 'allocation': 30},
                    {'name': '해외 성장주', 'allocation': 30},
                    {'name': '대체 투자', 'allocation': 20},
                    {'name': '채권', 'allocation': 10},
                    {'name': '현금', 'allocation': 10}
                ]
            }]
        elif risk >= -0.5:
            portfolio = [{
                'name': '균형형 포트폴리오',
                'description': '위험과 수익의 균형을 추구하는 포트폴리오',
                'assets': [
                    {'name': '국내 주식', 'allocation': 25},
                    {'name': '해외 주식', 'allocation': 20},
                    {'name': '채권', 'allocation': 35},
                    {'name': '대체 투자', 'allocation': 10},
                    {'name': '현금', 'allocation': 10}
                ]
            }]
        else:
            portfolio = [{
                'name': '안정형 포트폴리오',
                'description': '안정성과 원금 보존을 중시하는 포트폴리오',
                'assets': [
                    {'name': '국내 채권', 'allocation': 50},
                    {'name': '해외 채권', 'allocation': 20},
                    {'name': '국내 주식', 'allocation': 10},
                    {'name': '현금', 'allocation': 20}
                ]
            }]
        
        return {
            'portfolio': portfolio,
            'reason': reason_text
        }
    
    def _save_to_file(self, result):
        """분석 결과를 파일에 저장"""
        try:
            with open(self.analysis_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"분석 결과가 저장되었습니다: {self.analysis_file}")
        except Exception as e:
            print(f"분석 결과 저장 중 오류 발생: {e}")