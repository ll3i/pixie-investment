#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flask 라우트 확인"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from app import app

print("등록된 라우트 목록:")
print("=" * 60)

# 모든 라우트 출력
for rule in app.url_map.iter_rules():
    methods = ','.join(rule.methods - {'HEAD', 'OPTIONS'})
    print(f"{rule.endpoint:50s} {methods:8s} {rule}")

print("\n" + "=" * 60)

# evaluation-insights 라우트 확인
evaluation_route_found = False
for rule in app.url_map.iter_rules():
    if 'evaluation-insights' in str(rule):
        evaluation_route_found = True
        print(f"\n✓ evaluation-insights 라우트 발견: {rule}")
        break

if not evaluation_route_found:
    print("\n✗ evaluation-insights 라우트를 찾을 수 없습니다!")
else:
    print("\n라우트가 정상적으로 등록되어 있습니다.")
    print("Flask 서버를 재시작하면 API를 사용할 수 있습니다.")