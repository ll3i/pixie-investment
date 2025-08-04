"""
Python 내장 HTTP 서버를 사용한 대체 방법
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json

# Python path 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

class FlaskProxyHandler(SimpleHTTPRequestHandler):
    """Flask 앱을 프록시하는 핸들러"""
    
    def do_GET(self):
        """GET 요청 처리"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # 간단한 HTML 페이지
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pixie 투자챗봇</title>
                <meta charset="utf-8">
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    .container {
                        text-align: center;
                        margin-top: 50px;
                    }
                    .btn {
                        display: inline-block;
                        padding: 10px 20px;
                        margin: 10px;
                        background-color: #007bff;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                    }
                    .btn:hover {
                        background-color: #0056b3;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 Pixie 투자챗봇</h1>
                    <p>AI 기반 맞춤형 투자 조언 서비스</p>
                    
                    <div style="margin-top: 30px;">
                        <h2>서비스 상태</h2>
                        <p>✅ 서버가 정상적으로 실행 중입니다.</p>
                        <p>Flask 서버가 소켓 오류로 실행되지 않아 임시 페이지를 표시합니다.</p>
                    </div>
                    
                    <div style="margin-top: 30px;">
                        <h2>문제 해결 방법</h2>
                        <ol style="text-align: left; display: inline-block;">
                            <li>관리자 권한으로 <code>fix_socket_issue.bat</code> 실행</li>
                            <li>컴퓨터 재시작</li>
                            <li>Windows Defender 일시 중지</li>
                            <li>다른 포트로 시도: <code>python start_server_simple.py</code></li>
                        </ol>
                    </div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_error(404, "Page not found")

def start_http_server(port=8080):
    """HTTP 서버 시작"""
    print(f"임시 HTTP 서버를 시작합니다...")
    print(f"브라우저에서 http://localhost:{port} 으로 접속하세요")
    print("종료하려면 Ctrl+C를 누르세요\n")
    
    try:
        server = HTTPServer(('127.0.0.1', port), FlaskProxyHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    except Exception as e:
        print(f"서버 실행 중 오류: {e}")

if __name__ == '__main__':
    start_http_server()