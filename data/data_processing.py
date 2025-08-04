#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Processing Script - 한국 주식 데이터 수집 및 처리
이 스크립트는 Jupyter 노트북을 Python 스크립트로 변환한 것입니다.
"""

import sys
from bs4 import BeautifulSoup
import requests as rq
from io import BytesIO
import pandas as pd
import numpy as np
import time
import re
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from datetime import date
import csv
import math
from collections import Counter
import pickle
import os

print("데이터 처리 스크립트를 시작합니다...")

# 1. Data Crawling
print("\n=== 1. 데이터 크롤링 시작 ===")

# 1-2. 영업일 가져오기
def biz_day():
    """현재 영업일을 가져오는 함수"""
    url = 'https://finance.naver.com/sise/sise_deposit.nhn'
    data = rq.get(url)
    data_html = BeautifulSoup(data.content, features='lxml')
    parse_day = data_html.select_one('div.subtop_sise_graph2 > ul.subtop_chart_note > li > span.tah').text

    biz_day = re.findall('[0-9]+', parse_day)
    biz_day = ''.join(biz_day)

    return biz_day

# 영업일 가져오기
try:
    bday = biz_day()
    print(f"현재 영업일: {bday}")
except Exception as e:
    print(f"영업일을 가져오는 중 오류 발생: {e}")
    # 오류 시 오늘 날짜 사용
    bday = date.today().strftime("%Y%m%d")
    print(f"오늘 날짜를 사용합니다: {bday}")

# 1-3. Ticker List
print("\n=== 종목 리스트 다운로드 ===")
gen_otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'

# 코스피 params
gen_otp_stk = {
    'mktId': 'STK',
    'trdDd': bday,
    'money': '1',
    'csvxls_isNo': 'false',
    'name': 'fileDown',
    'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
}

# 경로 우회 headers
headers = {'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader',
           'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'}

try:
    # 코스피 post
    otp_stk = rq.post(gen_otp_url, gen_otp_stk, headers=headers).text
    time.sleep(1)

    # csv download
    down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'

    # 코스피 다운로드
    down_sector_stk = rq.post(down_url, {'code': otp_stk}, headers=headers)
    time.sleep(1)

    # 섹터 정보
    sector_stk = pd.read_csv(BytesIO(down_sector_stk.content), encoding='EUC-KR')

    # 코스닥 params
    gen_otp_ksq = {
        'mktId': 'KSQ',
        'trdDd': bday,
        'money': '1',
        'csvxls_isNo': 'false',
        'name': 'fileDown',
        'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
    }

    # 코스닥 post
    otp_ksq = rq.post(gen_otp_url, gen_otp_ksq, headers=headers).text
    time.sleep(1)

    # 코스닥 다운로드
    down_sector_ksq = rq.post(down_url, {'code': otp_ksq}, headers=headers)
    time.sleep(1)

    # 섹터 정보
    sector_ksq = pd.read_csv(BytesIO(down_sector_ksq.content), encoding='EUC-KR')

    # 병합
    krx_sector = pd.concat([sector_stk, sector_ksq]).reset_index(drop=True)

    # 데이터 전처리
    krx_sector['종목명'] = krx_sector['종목명'].str.strip()
    krx_sector['기준일'] = bday

    # ind params
    gen_otp_data = {
        'searchType': '1',
        'mktId': 'ALL',
        'trdDd': bday,
        'csvxls_isNo': 'false',
        'name': 'fileDown',
        'url': 'dbms/MDC/STAT/standard/MDCSTAT03501'
    }

    otp = rq.post(gen_otp_url, gen_otp_data, headers=headers).text
    time.sleep(1)

    # 산업별 현황
    krx_ind = rq.post(down_url, {'code': otp}, headers=headers)
    time.sleep(1)
    krx_ind = pd.read_csv(BytesIO(krx_ind.content), encoding='EUC-KR')

    krx_ind['종목명'] = krx_ind['종목명'].str.strip()
    krx_ind['기준일'] = bday

    # 대칭 차집합 반환
    diff = list(set(krx_sector['종목명']).symmetric_difference(set(krx_ind['종목명'])))

    kor_ticker = pd.merge(krx_sector,
                        krx_ind,
                        on=krx_sector.columns.intersection(krx_ind.columns).tolist(),
                        how='outer')

    # 종목 스팩 우선주 리츠 기타 주식 구분 작업
    kor_ticker['종목구분'] = np.where(kor_ticker['종목명'].str.contains('스팩|제[0-9]+호'), '스팩',
                                np.where(kor_ticker['종목코드'].str[-1:] != '0', '우선주',
                                        np.where(kor_ticker['종목명'].str.endswith('리츠'), '리츠',
                                                    np.where(kor_ticker['종목명'].isin(diff),  '기타',
                                                    '보통주'))))
    kor_ticker = kor_ticker.reset_index(drop=True)

    # 공백 제거
    kor_ticker.columns = kor_ticker.columns.str.replace(' ', '')

    # 필요한 정보 슬라이싱
    kor_ticker = kor_ticker[['종목코드', '종목명', '시장구분', '종가',
                            '시가총액', '기준일', 'EPS', '선행EPS', 'BPS', '주당배당금', '종목구분']]

    kor_ticker['종목코드'] = kor_ticker['종목코드'].astype(str).str.zfill(6)

    # CSV 파일로 저장
    output_filename = f'kor_ticker_{bday}.csv'
    kor_ticker.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    print(f'Ticker List Downloaded and saved as {output_filename}')
except Exception as e:
    print(f"Ticker List 다운로드 중 오류 발생: {e}")

# 1-4. Sector List
print("\n=== 섹터 리스트 다운로드 ===")
sector_code = [
    'G25', 'G35', 'G50', 'G40', 'G10', 'G20', 'G55', 'G30', 'G15', 'G45'
]

try:
    data_sector = []
    for i in tqdm(sector_code):
        url = f'''http://www.wiseindex.com/Index/GetIndexComponets?ceil_yn=0&dt={bday}&sec_cd={i}'''    
        data = rq.get(url).json()
        data_pd = pd.json_normalize(data['list'])
        data_sector.append(data_pd)
        time.sleep(2)

    kor_sector = pd.concat(data_sector, axis=0)
    kor_sector = kor_sector[['IDX_CD', 'CMP_CD', 'CMP_KOR', 'SEC_NM_KOR']]
    kor_sector['기준일'] = bday
    kor_sector['기준일'] = pd.to_datetime(kor_sector['기준일'])

    # CMP_CD(종목코드)를 문자열로 변환하고 6자리로 맞춤
    kor_sector['CMP_CD'] = kor_sector['CMP_CD'].astype(str).str.zfill(6)

    # CSV 파일로 저장
    output_filename = f'kor_sector_{bday}.csv'
    kor_sector.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    print(f'Sector List Downloaded and saved as {output_filename}')
except Exception as e:
    print(f"Sector List 다운로드 중 오류 발생: {e}")

# 1-5. Price List (테스트용으로 30개만)
print("\n=== 주가 리스트 다운로드 (30개 종목만) ===")
try:
    ticker_list = pd.read_csv(f'kor_ticker_{bday}.csv', dtype={'종목코드': str})
    ticker_list = ticker_list[ticker_list['종목구분'] == '보통주']

    error_list = []

    # 테스트를 위해 처음 30개의 종목만 선택
    ticker_list = ticker_list[:30]

    # 전체 가격 데이터를 저장할 DataFrame
    all_prices = pd.DataFrame()

    # 전종목 주가 다운로드 및 저장
    for i in tqdm(range(0, len(ticker_list))):
        # 티커 선택
        ticker = ticker_list['종목코드'].iloc[i]

        # 시작일과 종료일
        fr = (date.today() + relativedelta(years=-5)).strftime("%Y%m%d")
        to = date.today().strftime("%Y%m%d")

        # 오류 발생 시 이를 무시하고 다음 루프로 진행
        try:
            # url 생성
            url = f'''https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}&requestType=1&startTime={fr}&endTime={to}&timeframe=day'''

            # 데이터 다운로드
            data = rq.get(url).content
            data_price = pd.read_csv(BytesIO(data))

            # 데이터 클렌징
            price = data_price.iloc[:, 0:6]
            price.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
            price = price.dropna()
            price['날짜'] = price['날짜'].str.extract('(\d+)')
            price['날짜'] = pd.to_datetime(price['날짜'])
            price['종목코드'] = ticker

            # 전체 가격 데이터에 추가
            all_prices = pd.concat([all_prices, price], ignore_index=True)
            
            print(f'\n{ticker} is processed.\n')

        except:
            # 오류 발생시 error_list에 티커 저장하고 넘어가기
            print(f"\nYou've got an error on {ticker}")
            error_list.append(ticker)

        print(f'{len(error_list)} error(s) occurred now.\n')
        # 타임 슬립
        time.sleep(2)

    # 종목코드를 문자열로 변환하고 6자리로 맞춤
    all_prices['종목코드'] = all_prices['종목코드'].astype(str).str.zfill(6)

    # CSV 파일로 저장
    output_filename = f'kor_price_{bday}.csv'
    all_prices.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    print(f'Price List Downloaded and saved as {output_filename}')
    print(f'Total {len(error_list)} errors occurred.')
    if error_list:
        print("Tickers with errors:", error_list)
except Exception as e:
    print(f"Price List 다운로드 중 오류 발생: {e}")

# 1-6. Financial Statement List (테스트용으로 30개만)
print("\n=== 재무제표 리스트 다운로드 (30개 종목만) ===")
try:
    # CSV 파일에서 티커 리스트 읽기
    ticker_list = pd.read_csv(f'kor_ticker_{bday}.csv', dtype={'종목코드': str})
    ticker_list = ticker_list[ticker_list['종목구분'] == '보통주']

    ticker_list = ticker_list[:30]

    error_list = []

    # 재무제표 클렌징 함수
    def clean_fs(df, ticker, frequency):
        df = df[~df.loc[:, ~df.columns.isin(['계정'])].isna().all(axis=1)]
        df = df.drop_duplicates(['계정'], keep='first')
        df = pd.melt(df, id_vars='계정', var_name='기준일', value_name='값')
        df = df[~pd.isnull(df['값'])]
        df['계정'] = df['계정'].replace({'계산에 참여한 계정 펼치기': ''}, regex=True)
        df['기준일'] = pd.to_datetime(df['기준일'], format='%Y/%m') + pd.tseries.offsets.MonthEnd()
        df['종목코드'] = ticker
        df['공시구분'] = frequency
        return df

    # 전체 재무제표 데이터를 저장할 DataFrame
    all_fs_data = pd.DataFrame()

    # for loop
    for i in tqdm(range(0, len(ticker_list))):
        # 티커 선택
        ticker = ticker_list['종목코드'].iloc[i]

        # 오류 발생 시 이를 무시하고 다음 루프로 진행
        try:
            # url 생성
            url = f'http://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A{ticker}'

            # 데이터 받아오기
            data = pd.read_html(url, displayed_only=False)

            # 연간 데이터
            data_fs_y = pd.concat([
                data[0].iloc[:, ~data[0].columns.str.contains('전년동기')], data[2],
                data[4]
            ])
            data_fs_y = data_fs_y.rename(columns={data_fs_y.columns[0]: "계정"})

            # 결산년 찾기
            page_data = rq.get(url)
            page_data_html = BeautifulSoup(page_data.content, 'html.parser')

            fiscal_data = page_data_html.select('div.corp_group1 > h2')
            fiscal_data_text = fiscal_data[1].text
            fiscal_data_text = re.findall('[0-9]+', fiscal_data_text)

            # 결산년에 해당하는 계정만 남기기
            data_fs_y = data_fs_y.loc[:, (data_fs_y.columns == '계정') | (
                data_fs_y.columns.str[-2:].isin(fiscal_data_text))]

            # 클렌징
            data_fs_y_clean = clean_fs(data_fs_y, ticker, 'y')

            # 분기 데이터
            data_fs_q = pd.concat([
                data[1].iloc[:, ~data[1].columns.str.contains('전년동기')], data[3],
                data[5]
            ])
            data_fs_q = data_fs_q.rename(columns={data_fs_q.columns[0]: "계정"})

            data_fs_q_clean = clean_fs(data_fs_q, ticker, 'q')

            # 두개 합치기
            data_fs_bind = pd.concat([data_fs_y_clean, data_fs_q_clean])

            # 전체 재무제표 데이터에 추가
            all_fs_data = pd.concat([all_fs_data, data_fs_bind], ignore_index=True)

            print(f'\n{ticker} is processed.\n')

        except:
            # 오류 발생시 해당 종목명을 저장하고 다음 루프로 이동
            print(f"\nYou've got an error on {ticker}")
            error_list.append(ticker)
        
        print(f'{len(error_list)} error(s) occurred now.\n')
        # 타임슬립 적용
        time.sleep(2)

    # 종목코드를 문자열로 변환하고 6자리로 맞춤
    all_fs_data['종목코드'] = all_fs_data['종목코드'].astype(str).str.zfill(6)

    # CSV 파일로 저장
    output_filename = f'kor_fs_{bday}.csv'
    all_fs_data.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    print(f'Financial Statement List Downloaded and saved as {output_filename}')
    print(f'Total {len(error_list)} errors occurred.')
    if error_list:
        print("Tickers with errors:", error_list)
except Exception as e:
    print(f"Financial Statement List 다운로드 중 오류 발생: {e}")

# 1-7. Value List
print("\n=== 가치 지표 계산 ===")
try:
    kor_fs = pd.read_csv(f'kor_fs_{bday}.csv', dtype={'종목코드': str})
    kor_fs = kor_fs[
        (kor_fs['공시구분'] == 'q') & 
        (kor_fs['계정'].isin(['당기순이익', '자본', '영업활동으로인한현금흐름', '매출액']))
    ]

    # CSV 파일에서 티커 리스트 읽기
    ticker_list = pd.read_csv(f'kor_ticker_{bday}.csv', dtype={'종목코드': str})
    ticker_list = ticker_list[ticker_list['종목구분'] == '보통주']

    # TTM 구하기
    kor_fs = kor_fs.sort_values(['종목코드', '계정', '기준일'])
    kor_fs['ttm'] = kor_fs.groupby(['종목코드', '계정'], as_index=False)['값'].rolling(
        window=4, min_periods=4).sum()['값']

    # 자본은 평균 구하기
    kor_fs['ttm'] = np.where(kor_fs['계정'] == '자본', kor_fs['ttm'] / 4, kor_fs['ttm'])
    kor_fs = kor_fs.groupby(['계정', '종목코드']).tail(1)

    kor_fs_merge = kor_fs[['계정', '종목코드', 'ttm']].merge(
        ticker_list[['종목코드', '시가총액', '기준일']], on='종목코드')
    kor_fs_merge['시가총액'] = kor_fs_merge['시가총액'] / 100000000

    kor_fs_merge['value'] = kor_fs_merge['시가총액'] / kor_fs_merge['ttm']
    kor_fs_merge['value'] = kor_fs_merge['value'].round(4)
    kor_fs_merge['지표'] = np.where(
        kor_fs_merge['계정'] == '매출액', 'PSR',
        np.where(
            kor_fs_merge['계정'] == '영업활동으로인한현금흐름', 'PCR',
            np.where(kor_fs_merge['계정'] == '자본', 'PBR',
                    np.where(kor_fs_merge['계정'] == '당기순이익', 'PER', None))))

    kor_fs_merge.rename(columns={'value': '값'}, inplace=True)
    kor_fs_merge = kor_fs_merge[['종목코드', '기준일', '지표', '값']]
    kor_fs_merge = kor_fs_merge.replace([np.inf, -np.inf, np.nan], None)

    # 배당수익률 계산
    ticker_list['값'] = ticker_list['주당배당금'] / ticker_list['종가']
    ticker_list['값'] = ticker_list['값'].round(4)
    ticker_list['지표'] = 'DY'
    dy_list = ticker_list[['종목코드', '기준일', '지표', '값']]
    dy_list = dy_list.replace([np.inf, -np.inf, np.nan], None)
    dy_list = dy_list[dy_list['값'] != 0]

    # 모든 밸류 데이터 합치기
    all_value_data = pd.concat([kor_fs_merge, dy_list], ignore_index=True)

    # 종목코드를 문자열로 변환하고 6자리로 맞춤
    all_value_data['종목코드'] = all_value_data['종목코드'].astype(str).str.zfill(6)

    # CSV 파일로 저장
    output_filename = f'kor_value_{bday}.csv'
    all_value_data.to_csv(output_filename, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    print(f'Value List Calculated and saved as {output_filename}')
except Exception as e:
    print(f"Value List 계산 중 오류 발생: {e}")

# 2. Stock Evaluation
print("\n\n=== 2. 주식 평가 시작 ===")

# 2-2. 주식 평가 함수 정의
def evaluate_stock(stock_code, fs_data, price_data, ticker_data, value_data):
    print(f"평가 시작: 종목 코드 {stock_code}")
    
    # 종목 정보 추출
    stock_info = ticker_data[ticker_data['종목코드'] == stock_code].iloc[0]
    print(f"종목명: {stock_info['종목명']}")

    # 재무 데이터 추출
    fs = fs_data[(fs_data['종목코드'] == stock_code) & (fs_data['공시구분'] == 'y')].sort_values('기준일', ascending=False)
    
    # 주가 데이터 추출
    price = price_data[price_data['종목코드'] == stock_code].sort_values('날짜', ascending=False)
    
    # 투자 지표 추출
    value = value_data[value_data['종목코드'] == stock_code]

    score = 0
    reasons = []

    # 성장성 평가
    try:
        revenue_current = fs[fs['계정'] == '매출액']['값'].iloc[0]
        revenue_previous = fs[fs['계정'] == '매출액']['값'].iloc[1]
        revenue_growth = (revenue_current - revenue_previous) / revenue_previous * 100
        print(f"매출 성장률: {revenue_growth:.2f}%")
        if revenue_growth > 10:
            score += 20
            reasons.append(f"매출 성장률이 {revenue_growth:.2f}%로 우수함")
        elif revenue_growth > 0:
            score += 10
            reasons.append(f"매출 성장률이 {revenue_growth:.2f}%로 양호함")
        else:
            reasons.append(f"매출 성장률이 {revenue_growth:.2f}%로 저조함")
    except Exception as e:
        print(f"성장성 평가 중 오류 발생: {e}")
        revenue_growth = np.nan

    # 수익성 평가
    try:
        net_profit = fs[fs['계정'] == '당기순이익']['값'].iloc[0]
        revenue = fs[fs['계정'] == '매출액']['값'].iloc[0]
        net_profit_margin = net_profit / revenue * 100
        print(f"순이익률: {net_profit_margin:.2f}%")
        if net_profit_margin > 10:
            score += 20
            reasons.append(f"순이익률이 {net_profit_margin:.2f}%로 우수함")
        elif net_profit_margin > 5:
            score += 10
            reasons.append(f"순이익률이 {net_profit_margin:.2f}%로 양호함")
        else:
            reasons.append(f"순이익률이 {net_profit_margin:.2f}%로 개선 필요")
    except Exception as e:
        print(f"수익성 평가 중 오류 발생: {e}")
        net_profit_margin = np.nan

    # 재무 안정성 평가
    try:
        debt = fs[fs['계정'] == '부채']['값'].iloc[0]
        equity = fs[fs['계정'] == '자본']['값'].iloc[0]
        debt_ratio = debt / equity * 100
        print(f"부채비율: {debt_ratio:.2f}%")
        if debt_ratio < 100:
            score += 20
            reasons.append(f"부채비율이 {debt_ratio:.2f}%로 안정적임")
        elif debt_ratio < 200:
            score += 10
            reasons.append(f"부채비율이 {debt_ratio:.2f}%로 보통 수준")
        else:
            reasons.append(f"부채비율이 {debt_ratio:.2f}%로 높음")
    except Exception as e:
        print(f"재무 안정성 평가 중 오류 발생: {e}")
        debt_ratio = np.nan

    # 투자 지표 평가
    try:
        per = float(value[value['지표'] == 'PER']['값'])
        pbr = float(value[value['지표'] == 'PBR']['값'])
        print(f"PER: {per:.2f}")
        print(f"PBR: {pbr:.2f}")

        if 5 < per < 15:
            score += 20
            reasons.append(f"PER이 {per:.2f}로 적정 수준")
        elif per <= 5:
            score += 10
            reasons.append(f"PER이 {per:.2f}로 저평가 가능성")
        else:
            reasons.append(f"PER이 {per:.2f}로 고평가 가능성")

        if 0.5 < pbr < 2:
            score += 20
            reasons.append(f"PBR이 {pbr:.2f}로 적정 수준")
        elif pbr <= 0.5:
            score += 10
            reasons.append(f"PBR이 {pbr:.2f}로 저평가 가능성")
        else:
            reasons.append(f"PBR이 {pbr:.2f}로 고평가 가능성")
    except Exception as e:
        print(f"투자 지표 평가 중 오류 발생: {e}")
        per, pbr = np.nan, np.nan

    # 종합 평가
    if score >= 80:
        evaluation = "매우 좋음"
    elif score >= 60:
        evaluation = "좋음"
    elif score >= 40:
        evaluation = "보통"
    else:
        evaluation = "주의 필요"

    result = {
        "종목명": stock_info['종목명'],
        "종목코드": stock_code,
        "현재가": stock_info['종가'],
        "시가총액": stock_info['시가총액'],
        "매출성장률": revenue_growth,
        "순이익률": net_profit_margin,
        "부채비율": debt_ratio,
        "PER": per,
        "PBR": pbr,
        "평가점수": score,
        "종합평가": evaluation,
        "평가이유": '; '.join(reasons)
    }

    return result

# 2-3. 파일 불러오기
print("\n=== 평가를 위한 데이터 로드 ===")
try:
    # 데이터 로드 (이 부분은 함수 외부에서 한 번만 실행)
    fs_data = pd.read_csv(f'kor_fs_{bday}.csv', dtype={'종목코드': str}).fillna({'종목코드': '000000'})
    fs_data['종목코드'] = fs_data['종목코드'].str.zfill(6)

    price_data = pd.read_csv(f'kor_price_{bday}.csv', dtype={'종목코드': str}).fillna({'종목코드': '000000'})
    price_data['종목코드'] = price_data['종목코드'].str.zfill(6)

    ticker_data = pd.read_csv(f'kor_ticker_{bday}.csv', dtype={'종목코드': str}).fillna({'종목코드': '000000'})
    ticker_data['종목코드'] = ticker_data['종목코드'].str.zfill(6)

    value_data = pd.read_csv(f'kor_value_{bday}.csv', dtype={'종목코드': str}).fillna({'종목코드': '000000'})
    value_data['종목코드'] = value_data['종목코드'].str.zfill(6)

    print("데이터 로드 완료")
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")

# 2-4. 함수 실행
print("\n=== 주식 평가 실행 ===")
try:
    # 결과를 저장할 리스트 생성
    results = []

    # 함수 사용 예시
    stock_codes = ticker_data['종목코드'].unique()
    for code in stock_codes:
        result = evaluate_stock(code, fs_data, price_data, ticker_data, value_data)
        results.append(result)
        print("\n===== 평가 결과 =====")
        for key, value in result.items():
            print(f"{key}: {value}")
        print("\n")

    # 2-5. 결과 저장
    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)

    # 데이터프레임을 CSV 파일로 저장
    results_df.to_csv('stock_evaluation_results.csv', index=False, encoding='utf-8-sig')
    print("평가 결과가 'stock_evaluation_results.csv' 파일로 저장되었습니다.")
except Exception as e:
    print(f"주식 평가 중 오류 발생: {e}")

# 3. Data Embedding
print("\n\n=== 3. 데이터 임베딩 시작 ===")

# 3-2. 데이터 준비
try:
    df = pd.read_csv('stock_evaluation_results.csv')
    df = df[df['평가점수'] != 0]  # 평가점수가 0인 항목 제거
    df.reset_index(drop=True, inplace=True)  # 인덱스 재설정
    print(f"임베딩을 위한 데이터 준비 완료: {len(df)}개 종목")
except Exception as e:
    print(f"데이터 준비 중 오류 발생: {e}")

# 3-3. 텍스트 데이터 생성
def create_text_for_embedding(row):
    text = f"종목명: {row['종목명']}, 종목코드: {row['종목코드']}, "
    text += f"현재가: {row['현재가']}, 시가총액: {row['시가총액']}, "
    text += f"매출성장률: {row['매출성장률']}, 순이익률: {row['순이익률']}, "
    text += f"부채비율: {row['부채비율']}, PER: {row['PER']}, PBR: {row['PBR']}, "
    text += f"평가점수: {row['평가점수']}, 종합평가: {row['종합평가']}, "
    text += f"평가이유: {row['평가이유']}"
    return text

try:
    df['text'] = df.apply(create_text_for_embedding, axis=1)
    print("텍스트 데이터 생성 완료")
except Exception as e:
    print(f"텍스트 데이터 생성 중 오류 발생: {e}")

# 3-4. 간단한 벡터화 함수
def tokenize(text):
    return re.findall(r'\w+', text.lower())

def compute_tf(text):
    tf_dict = Counter(tokenize(text))
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / float(len(tf_dict))
    return tf_dict

def compute_idf(corpus):
    idf_dict = {}
    N = len(corpus)
    
    all_words = set(word for text in corpus for word in tokenize(text))
    
    for word in all_words:
        count = sum(1 for text in corpus if word in tokenize(text))
        idf_dict[word] = math.log(N / float(count))
    
    return idf_dict

def compute_tfidf(tf, idf):
    tfidf = {}
    for word, tf_value in tf.items():
        tfidf[word] = tf_value * idf.get(word, 0)
    return tfidf

# 3-5. 벡터화 및 데이터 저장
try:
    corpus = df['text'].tolist()
    idf = compute_idf(corpus)

    vectors = []
    for text in corpus:
        tf = compute_tf(text)
        tfidf = compute_tfidf(tf, idf)
        vectors.append(tfidf)

    # 데이터 프레임과 벡터, IDF 저장
    df['vector'] = vectors
    df.to_pickle('stock_data.pkl')
    with open('idf.pkl', 'wb') as f:
        pickle.dump(idf, f)
    print("벡터화 완료 및 데이터 저장 완료")
    print(df)
except Exception as e:
    print(f"벡터화 중 오류 발생: {e}")

# 3-7. 검색 함수
def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def search_stocks(query, n_results=5):
    # 데이터 프레임과 IDF 로드
    df = pd.read_pickle('stock_data.pkl')
    with open('idf.pkl', 'rb') as f:
        idf = pickle.load(f)
    
    # 쿼리 벡터화
    query_tf = compute_tf(query)
    query_vector = compute_tfidf(query_tf, idf)
    
    # 유사도 계산 및 정렬
    similarities = [(i, cosine_similarity(query_vector, row['vector'])) 
                    for i, row in df.iterrows()]
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, _ in similarities[:n_results]:
        stock_info = df.iloc[idx]
        result = f"종목명: {stock_info['종목명']}, 종목코드: {stock_info['종목코드']}, "
        result += f"현재가: {stock_info['현재가']}, 시가총액: {stock_info['시가총액']}, "
        result += f"매출성장률: {stock_info['매출성장률']}, 순이익률: {stock_info['순이익률']}, "
        result += f"부채비율: {stock_info['부채비율']}, PER: {stock_info['PER']}, PBR: {stock_info['PBR']}, "
        result += f"평가점수: {stock_info['평가점수']}, 종합평가: {stock_info['종합평가']}"
        results.append(result)
    
    return results

print("\n스크립트 실행 완료!")
print(f"\n생성된 파일들:")
print(f"- kor_ticker_{bday}.csv")
print(f"- kor_sector_{bday}.csv")
print(f"- kor_price_{bday}.csv")
print(f"- kor_fs_{bday}.csv")
print(f"- kor_value_{bday}.csv")
print(f"- stock_evaluation_results.csv")
print(f"- stock_data.pkl")
print(f"- idf.pkl")

# API를 사용하려면 다음과 같이 할 수 있습니다:
# 1. LLM API 설정 (OpenAI, Clova 등)
# 2. stock_chat 함수 구현
# 3. 웹 서비스로 통합