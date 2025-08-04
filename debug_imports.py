"""
임포트 문제 디버깅
"""
import sys
print("Python 버전:", sys.version)
print("\n모듈 임포트 테스트:")

try:
    import flask
    print("✓ Flask 임포트 성공")
except Exception as e:
    print(f"✗ Flask 임포트 실패: {e}")

try:
    import pandas
    print("✓ Pandas 임포트 성공")
except Exception as e:
    print(f"✗ Pandas 임포트 실패: {e}")

try:
    import numpy
    print("✓ Numpy 임포트 성공")
except Exception as e:
    print(f"✗ Numpy 임포트 실패: {e}")

try:
    import sklearn
    print("✓ Scikit-learn 임포트 성공")
except Exception as e:
    print(f"✗ Scikit-learn 임포트 실패: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ Sentence-transformers 임포트 성공")
except Exception as e:
    print(f"✗ Sentence-transformers 임포트 실패: {e}")

print("\n간단한 Flask 앱 실행 테스트:")
try:
    from flask import Flask
    test_app = Flask(__name__)
    
    @test_app.route('/')
    def test():
        return "테스트 성공!"
    
    print("Flask 앱 생성 성공. 실행 중...")
    test_app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
except Exception as e:
    print(f"Flask 실행 실패: {e}")