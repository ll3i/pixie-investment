// 챗봇 위젯 JavaScript
class ChatbotWidget {
    constructor() {
        this.isOpen = false;
        this.messages = [];
        this.sessionId = localStorage.getItem('session_id') || this.generateSessionId();
        this.init();
    }

    init() {
        this.createWidget();
        this.attachEventListeners();
        this.loadChatHistory();
        this.sendInitialMessage();
    }

    generateSessionId() {
        const sessionId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
        localStorage.setItem('session_id', sessionId);
        return sessionId;
    }

    createWidget() {
        const widgetHTML = `
            <div class="chatbot-widget">
                <button class="chatbot-trigger" id="chatbot-trigger">
                    <img src="/static/images/c1.png" alt="챗봇">
                </button>
                
                <div class="chatbot-backdrop" id="chatbot-backdrop"></div>
                
                <div class="chatbot-window" id="chatbot-window">
                    <div class="chatbot-header">
                        <div class="chatbot-header-info">
                            <h3>픽시 AI 어드바이저</h3>
                            <p>투자에 대한 모든 질문에 답변해드립니다</p>
                        </div>
                        <button class="chatbot-close" id="chatbot-close">×</button>
                    </div>
                    
                    <div class="chatbot-messages" id="chatbot-messages">
                        <!-- 메시지가 여기에 표시됩니다 -->
                    </div>
                    
                    <div class="chatbot-suggestions">
                        <button class="suggestion-btn" data-suggestion="나의 투자 성향에 어울리는 주식 추천해줘">투자 성향 기반 추천</button>
                        <button class="suggestion-btn" data-suggestion="포트폴리오 구성법">포트폴리오 구성</button>
                        <button class="suggestion-btn" data-suggestion="리스크 관리 방법">리스크 관리</button>
                        <button class="suggestion-btn" data-suggestion="오늘의 시장 분석">시장 분석</button>
                    </div>
                    
                    <div class="chatbot-action-buttons">
                        <button class="action-btn primary" id="survey-start-btn">
                            <i class="fas fa-clipboard-list"></i> 투자 설문 시작하기
                        </button>
                        <button class="action-btn secondary" id="survey-result-btn">
                            <i class="fas fa-chart-pie"></i> 설문 결과 보기
                        </button>
                    </div>
                    
                    <div class="chatbot-input-area">
                        <input 
                            type="text" 
                            class="chatbot-input" 
                            id="chatbot-input" 
                            placeholder="투자에 대해 궁금한 점을 물어보세요"
                            maxlength="500"
                        >
                        <button class="chatbot-send" id="chatbot-send">
                            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                                <path d="M19 1L9 11M19 1L13 19L9 11M19 1L1 7L9 11" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', widgetHTML);
    }

    attachEventListeners() {
        // 챗봇 열기/닫기
        document.getElementById('chatbot-trigger').addEventListener('click', () => this.toggleChatbot());
        document.getElementById('chatbot-close').addEventListener('click', () => this.closeChatbot());
        
        // 백드롭 클릭시 닫기
        document.getElementById('chatbot-backdrop').addEventListener('click', () => this.closeChatbot());

        // 메시지 전송
        document.getElementById('chatbot-send').addEventListener('click', () => this.sendMessage());
        document.getElementById('chatbot-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // 추천 버튼 클릭
        document.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const suggestion = e.target.getAttribute('data-suggestion');
                this.sendMessageText(suggestion);
            });
        });
        
        // ESC 키로 닫기
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.closeChatbot();
            }
        });
        
        // 설문 시작 버튼
        document.getElementById('survey-start-btn').addEventListener('click', () => {
            window.location.href = '/survey';
        });
        
        // 설문 결과 보기 버튼
        document.getElementById('survey-result-btn').addEventListener('click', (e) => {
            e.preventDefault();
            this.showSurveyResults();
        });
    }

    toggleChatbot() {
        this.isOpen = !this.isOpen;
        const window = document.getElementById('chatbot-window');
        const backdrop = document.getElementById('chatbot-backdrop');
        
        if (this.isOpen) {
            backdrop.style.display = 'block';
            // Force reflow to enable transition
            backdrop.offsetHeight;
            backdrop.classList.add('show');
            window.classList.add('show');
            document.getElementById('chatbot-input').focus();
            // 스크롤 방지
            document.body.style.overflow = 'hidden';
        } else {
            backdrop.classList.remove('show');
            window.classList.remove('show');
            setTimeout(() => {
                backdrop.style.display = 'none';
                document.body.style.overflow = '';
            }, 300);
        }
    }

    closeChatbot() {
        this.isOpen = false;
        const window = document.getElementById('chatbot-window');
        const backdrop = document.getElementById('chatbot-backdrop');
        
        backdrop.classList.remove('show');
        window.classList.remove('show');
        setTimeout(() => {
            backdrop.style.display = 'none';
            document.body.style.overflow = '';
        }, 300);
    }

    sendInitialMessage() {
        if (this.messages.length === 0) {
            this.addMessage('bot', '안녕하세요! 저는 투자 AI 파트너 픽시입니다.\n투자에 관한 모든 질문에 답변해드릴 수 있어요.');
            this.addMessage('bot', '어떤 도움이 필요하신가요?');
        }
    }

    async sendMessage() {
        const input = document.getElementById('chatbot-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        this.sendMessageText(message);
        input.value = '';
    }

    async sendMessageText(message) {
        // 사용자 메시지 추가
        this.addMessage('user', message);
        
        // 타이핑 인디케이터 표시
        this.showTypingIndicator();
        
        // 버튼 비활성화
        const sendBtn = document.getElementById('chatbot-send');
        const input = document.getElementById('chatbot-input');
        sendBtn.disabled = true;
        input.disabled = true;

        try {
            // EventSource for streaming updates
            const eventSource = new EventSource(`/api/chat-stream?message=${encodeURIComponent(message)}`);
            let fullResponse = '';
            let currentAgent = '';
            let agentResponses = {};
            
            eventSource.addEventListener('status', (event) => {
                const data = JSON.parse(event.data);
                if (data.agent && data.status === 'thinking') {
                    // Show AI thinking status
                    currentAgent = data.agent;
                }
            });
            
            // Agent response handler
            eventSource.addEventListener('agent_response', (event) => {
                const data = JSON.parse(event.data);
                if (data.agent && data.content) {
                    this.removeTypingIndicator();
                    // Add agent response
                    this.addAgentMessage(data.content, data.agent);
                    agentResponses[data.agent] = data.content;
                    // Show typing indicator for next agent
                    if (data.agent !== 'Final') {
                        this.showTypingIndicator();
                    }
                }
            });
            
            eventSource.addEventListener('message', (event) => {
                const data = JSON.parse(event.data);
                if (data.content) {
                    fullResponse += data.content;
                }
            });
            
            eventSource.addEventListener('complete', (event) => {
                this.removeTypingIndicator();
                eventSource.close();
                
                // If we have agent responses, add divider before final response
                if (Object.keys(agentResponses).length > 0 && fullResponse) {
                    // Add final response
                    this.addMessage('bot', fullResponse);
                } else if (Object.keys(agentResponses).length === 0 && fullResponse) {
                    // No agent responses, show the full response
                    this.addMessage('bot', fullResponse);
                }
                
                // 버튼 다시 활성화
                sendBtn.disabled = false;
                input.disabled = false;
                input.focus();
            });
            
            eventSource.addEventListener('error', (event) => {
                console.error('EventSource error:', event);
                eventSource.close();
                // Fallback to regular POST
                this.sendMessageFallback(message);
            });

        } catch (error) {
            console.error('Error:', error);
            this.sendMessageFallback(message);
        }
    }
    
    async sendMessageFallback(message) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Session-ID': this.sessionId
                },
                body: JSON.stringify({ 
                    message: message,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            
            this.removeTypingIndicator();
            
            if (data.success) {
                this.addMessage('bot', data.response);
            } else {
                this.addMessage('bot', '죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.');
            }
        } catch (error) {
            console.error('Error:', error);
            this.removeTypingIndicator();
            this.addMessage('bot', '죄송합니다. 네트워크 오류가 발생했습니다. 다시 시도해주세요.');
        } finally {
            // 버튼 다시 활성화
            const sendBtn = document.getElementById('chatbot-send');
            const input = document.getElementById('chatbot-input');
            sendBtn.disabled = false;
            input.disabled = false;
            input.focus();
        }
    }

    addMessage(sender, text) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chatbot-message ${sender}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        // HTML 콘텐츠를 지원하기 위해 innerHTML 사용
        if (sender === 'bot' && text.includes('<')) {
            bubble.innerHTML = text;
        } else {
            bubble.textContent = text;
        }
        
        messageDiv.appendChild(bubble);
        messagesContainer.appendChild(messageDiv);
        
        // 스크롤을 맨 아래로
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // 메시지 저장
        this.messages.push({ sender, text, timestamp: new Date() });
        this.saveChatHistory();
    }
    
    addAgentMessage(content, agent) {
        const messagesContainer = document.getElementById('chatbot-messages');
        const messageDiv = document.createElement('div');
        
        // 에이전트에 따른 클래스 설정
        let agentClass = '';
        let agentLabel = '';
        let agentIcon = '';
        
        if (agent === 'AI-A2') {
            agentClass = 'agent-a2';
            agentLabel = 'AI-A2 (투자 분석가)';
            agentIcon = 'A2';
        } else if (agent === 'AI-B') {
            agentClass = 'agent-b';
            agentLabel = 'AI-B (금융 데이터 전문가)';
            agentIcon = 'B';
        } else {
            agentClass = '';
            agentLabel = agent;
            agentIcon = 'AI';
        }
        
        messageDiv.className = `chatbot-message bot ${agentClass}`;
        messageDiv.innerHTML = `
            <div class="agent-label" style="font-weight: 700; color: #1454FE; font-size: 13px; margin-bottom: 8px;">
                <span>${agentLabel}</span>
            </div>
            <div class="message-bubble">${content}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // 채팅 기록에 추가
        this.messages.push({ 
            sender: 'agent', 
            agent: agent,
            text: content, 
            timestamp: new Date() 
        });
        this.saveChatHistory();
    }

    showTypingIndicator() {
        const messagesContainer = document.getElementById('chatbot-messages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chatbot-message bot typing-message';
        typingDiv.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    removeTypingIndicator() {
        const typingMessage = document.querySelector('.typing-message');
        if (typingMessage) {
            typingMessage.remove();
        }
    }

    saveChatHistory() {
        // 최근 50개 메시지만 저장
        const recentMessages = this.messages.slice(-50);
        localStorage.setItem('chatbot_messages', JSON.stringify(recentMessages));
    }

    loadChatHistory() {
        const savedMessages = localStorage.getItem('chatbot_messages');
        if (savedMessages) {
            try {
                this.messages = JSON.parse(savedMessages);
                // 저장된 메시지들을 화면에 표시
                this.messages.forEach(msg => {
                    const messagesContainer = document.getElementById('chatbot-messages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `chatbot-message ${msg.sender}`;
                    
                    const bubble = document.createElement('div');
                    bubble.className = 'message-bubble';
                    bubble.textContent = msg.text;
                    
                    messageDiv.appendChild(bubble);
                    messagesContainer.appendChild(messageDiv);
                });
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            } catch (e) {
                console.error('Failed to load chat history:', e);
                this.messages = [];
            }
        }
    }

    async showSurveyResults() {
        try {
            // 프로필 상태 확인
            const statusResponse = await fetch('/api/profile-status');
            const statusData = await statusResponse.json();
            
            if (!statusData.success || !statusData.has_profile) {
                this.addMessage('bot', '아직 투자 설문을 완료하지 않으셨습니다. 먼저 설문을 완료해주세요.');
                return;
            }
            
            // 프로필 데이터 가져오기
            const profileResponse = await fetch('/api/profile');
            const profileData = await profileResponse.json();
            
            if (profileData.success && profileData.profile) {
                const profile = profileData.profile;
                
                // 투자 성향 분석 결과 메시지 생성
                let resultMessage = '<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 15px;">';
                resultMessage += '<h3 style="color: #1454FE; margin-bottom: 20px;">📊 투자 성향 분석 결과</h3>';
                
                // Risk Tolerance
                resultMessage += '<div style="margin-bottom: 15px;">';
                resultMessage += '<h4 style="color: #333; margin-bottom: 5px;">Risk Tolerance</h4>';
                resultMessage += '<p style="color: #666; line-height: 1.6;">';
                resultMessage += `위험 감수성 점수가 <strong>${profile.risk_score || 0}점</strong>으로 나왔습니다. `;
                
                if (profile.risk_score >= 4) {
                    resultMessage += '이는 투자자가 높은 위험을 감수하는 것을 극도로 싫어한다는 것을 의미합니다. 이러한 성향은 안정적인 투자를 선호하게 만들지만, 높은 수익을 얻을 수 있는 기회를 놓치게 될 수도 있습니다.';
                } else if (profile.risk_score >= 2) {
                    resultMessage += '이는 투자자가 적절한 수준의 위험을 감수할 수 있다는 것을 의미합니다. 균형잡힌 포트폴리오 구성이 가능합니다.';
                } else {
                    resultMessage += '이는 투자자가 높은 위험을 기꺼이 감수한다는 것을 의미합니다. 고수익 고위험 투자에 적합하지만, 손실 위험도 큽니다.';
                }
                resultMessage += '</p></div>';
                
                // Investment Time Horizon
                resultMessage += '<div style="margin-bottom: 15px;">';
                resultMessage += '<h4 style="color: #333; margin-bottom: 5px;">Investment Time Horizon</h4>';
                resultMessage += '<p style="color: #666; line-height: 1.6;">';
                
                const timeScore = profile.answers?.q2 || 1;
                resultMessage += `투자 시간 범위 점수가 <strong>${timeScore === 1 ? '-1.0' : timeScore === 2 ? '0' : '1.0'}점</strong>으로 나왔습니다. `;
                
                if (timeScore <= 2) {
                    resultMessage += '이는 투자자가 단기적인 수익을 추구하는 경향이 있다는 것을 의미합니다. 단기적인 투자는 높은 위험을 감수해야 하며, 장기적인 투자에 비해 수익률이 낮을 가능성이 높습니다.';
                } else {
                    resultMessage += '이는 투자자가 장기적인 투자를 선호한다는 것을 의미합니다. 장기적인 투자는 시장 변동성을 극복하고 안정적인 수익을 얻을 수 있습니다.';
                }
                resultMessage += '</p></div>';
                
                // Financial Goal Orientation
                resultMessage += '<div style="margin-bottom: 15px;">';
                resultMessage += '<h4 style="color: #333; margin-bottom: 5px;">Financial Goal Orientation</h4>';
                resultMessage += '<p style="color: #666; line-height: 1.6;">';
                
                const goalScore = profile.answers?.q5 || 1;
                resultMessage += `재무 목표 지향성 점수가 <strong>${goalScore === 1 ? '-2.0' : goalScore === 2 ? '-1.0' : goalScore === 3 ? '0' : goalScore === 4 ? '1.0' : '2.0'}점</strong>으로 나왔습니다. `;
                
                resultMessage += '이는 투자자가 명확한 재무 목표를 가지고 있지 않을 가능성이 높다는 것을 의미합니다. 재무 목표가 명확하지 않으면 투자 결정에 어려움을 겪을 수 있으며, 투자 계획을 세우는 것도 어렵습니다. 재무 목표를 설정하고, 그에 맞는 투자 계획을 세우는 것이 중요합니다.';
                resultMessage += '</p></div>';
                
                // Information Processing Style
                resultMessage += '<div style="margin-bottom: 15px;">';
                resultMessage += '<h4 style="color: #333; margin-bottom: 5px;">Information Processing Style</h4>';
                resultMessage += '<p style="color: #666; line-height: 1.6;">';
                
                const infoScore = profile.answers?.q7 || 1;
                resultMessage += `정보 처리 스타일 점수가 <strong>${infoScore === 1 ? '-2.0' : infoScore === 2 ? '-1.0' : infoScore === 3 ? '0' : infoScore === 4 ? '1.0' : '2.0'}점</strong>으로 나왔습니다. `;
                
                resultMessage += '이는 투자자가 정보를 적극적으로 수집하고 분석하는 것을 어려워한다는 것을 의미합니다. 투자 결정을 내릴 때는 충분한 정보를 수집하고 분석하는 것이 중요합니다. 투자 관련 정보를 쉽게 이해할 수 있는 방법을 찾는 것이 좋습니다.';
                resultMessage += '</p></div>';
                
                // Investment Fear
                resultMessage += '<div style="margin-bottom: 15px;">';
                resultMessage += '<h4 style="color: #333; margin-bottom: 5px;">Investment Fear</h4>';
                resultMessage += '<p style="color: #666; line-height: 1.6;">';
                
                const fearScore = profile.answers?.q9 || 1;
                resultMessage += `투자 두려움 점수가 <strong>${fearScore === 1 ? '-2.0' : fearScore === 2 ? '-1.0' : fearScore === 3 ? '0' : fearScore === 4 ? '1.0' : '2.0'}점</strong>으로 나왔습니다. `;
                
                resultMessage += '이는 투자에 대한 두려움이 크지 않다는 것을 의미합니다. 하지만, 위험 감수성이 매우 낮기 때문에 적극적인 투자를 하기는 어려울 것입니다. 안정적인 자산에 투자하는 것이 좋습니다.';
                resultMessage += '</p></div>';
                
                // Investment Confidence
                resultMessage += '<div style="margin-bottom: 15px;">';
                resultMessage += '<h4 style="color: #333; margin-bottom: 5px;">Investment Confidence</h4>';
                resultMessage += '<p style="color: #666; line-height: 1.6;">';
                
                const confidenceScore = profile.answers?.q10 || 1;
                resultMessage += `투자 자신감 점수가 <strong>${confidenceScore === 1 ? '-3.0' : confidenceScore === 2 ? '-1.5' : confidenceScore === 3 ? '0' : confidenceScore === 4 ? '1.5' : '3.0'}점</strong>으로 나왔습니다. `;
                
                resultMessage += '이는 투자에 대한 자신감이 부족하다는 것을 의미합니다. 자신감이 부족하면 투자 결정을 내리는 것이 어려워지고, 투자 성과에 대한 만족도가 낮아질 수 있습니다. 자신감을 높이기 위해 투자 관련 지식을 습득하고, 투자 경험을 쌓는 것이 중요합니다.';
                resultMessage += '</p></div>';
                
                // Overall Evaluation
                resultMessage += '<div style="margin-bottom: 15px; background: #e7f0ff; padding: 15px; border-radius: 8px;">';
                resultMessage += '<h4 style="color: #1454FE; margin-bottom: 10px;">종합 평가</h4>';
                resultMessage += '<p style="color: #333; line-height: 1.6; margin-bottom: 10px;">';
                resultMessage += `귀하의 투자 성향은 <strong>${profile.personality_type || '분석 중'}</strong>입니다. `;
                
                if (profile.personality_type === '공격투자형') {
                    resultMessage += '높은 수익을 추구하며 위험을 감수할 수 있는 투자자입니다. 성장주, 테마주, 해외주식 등 고위험 고수익 상품에 적합합니다.';
                } else if (profile.personality_type === '적극투자형') {
                    resultMessage += '수익을 추구하면서도 일정 수준의 위험 관리를 하는 투자자입니다. 성장 가능성이 높은 우량주와 일부 성장주를 조합한 포트폴리오가 적합합니다.';
                } else if (profile.personality_type === '위험중립형') {
                    resultMessage += '안정성과 수익성의 균형을 추구하는 투자자입니다. 대형 우량주 중심으로 일부 성장주를 포함한 균형잡힌 포트폴리오가 적합합니다.';
                } else if (profile.personality_type === '안정추구형') {
                    resultMessage += '원금 보존을 중시하며 안정적인 수익을 추구하는 투자자입니다. 배당주, 대형 우량주 중심의 안정적인 포트폴리오가 적합합니다.';
                } else {
                    resultMessage += '원금 보존을 최우선으로 하는 보수적인 투자자입니다. 예금, 채권, 대형 우량주 위주의 매우 안정적인 포트폴리오가 적합합니다.';
                }
                
                resultMessage += '</p>';
                resultMessage += '<p style="color: #333; line-height: 1.6;">';
                resultMessage += '투자는 자신의 성향에 맞는 전략을 선택하는 것이 중요합니다. 위험을 너무 회피하면 수익 기회를 놓칠 수 있고, 과도한 위험을 감수하면 큰 손실을 볼 수 있습니다. 자신의 투자 성향을 잘 이해하고, 그에 맞는 투자 전략을 수립하시기 바랍니다.';
                resultMessage += '</p>';
                resultMessage += '</div>';
                
                resultMessage += '</div>';
                
                this.addMessage('bot', resultMessage);
                
                // 추가 안내 메시지
                setTimeout(() => {
                    this.addMessage('bot', '투자 성향을 바탕으로 구체적인 종목 추천을 원하시면 "내 투자 성향에 맞는 코스닥 종목 추천해줘" 라고 물어보세요!');
                }, 1000);
                
            } else {
                this.addMessage('bot', '투자 성향 정보를 불러오는 중 오류가 발생했습니다. 다시 시도해주세요.');
            }
        } catch (error) {
            console.error('설문 결과 표시 오류:', error);
            this.addMessage('bot', '설문 결과를 표시하는 중 오류가 발생했습니다.');
        }
    }
}

// DOM이 로드되면 챗봇 위젯 초기화
document.addEventListener('DOMContentLoaded', function() {
    new ChatbotWidget();
});