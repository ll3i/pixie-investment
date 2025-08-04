from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
app.secret_key = "test_key_123"

@app.route('/')
def index():
    return "Flask is working!"

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        # 간단한 테스트 응답
        return jsonify({
            "success": True,
            "response": f"테스트 응답: {message}"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    print("Starting test Flask server...")
    app.run(debug=True, port=5001)