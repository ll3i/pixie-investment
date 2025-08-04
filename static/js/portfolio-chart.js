// 포트폴리오 차트 기능
document.addEventListener('DOMContentLoaded', function() {
    // 차트가 있는 경우에만 실행
    const chartCanvas = document.getElementById('portfolioChart');
    if (!chartCanvas) return;

    // 차트 데이터 가져오기
    const portfolioData = window.portfolioData || {
        labels: ['국내 주식', '해외 주식', '채권', '현금성 자산', '대체 투자'],
        data: [35, 25, 20, 10, 10],
        stocks: {
            '국내 주식': [
                {code: '005930', name: '삼성전자', allocation: 15},
                {code: '000660', name: 'SK하이닉스', allocation: 10},
                {code: '035420', name: 'NAVER', allocation: 10}
            ],
            '해외 주식': [
                {code: 'AAPL', name: '애플', allocation: 10},
                {code: 'MSFT', name: '마이크로소프트', allocation: 8},
                {code: 'GOOGL', name: '구글', allocation: 7}
            ],
            '채권': [
                {code: 'KTB', name: '한국국채', allocation: 20}
            ],
            '현금성 자산': [
                {code: 'MMF', name: 'MMF', allocation: 10}
            ],
            '대체 투자': [
                {code: 'REIT', name: '리츠', allocation: 10}
            ]
        }
    };

    // Chart.js 설정
    const config = {
        type: 'pie',
        data: {
            labels: portfolioData.labels,
            datasets: [{
                data: portfolioData.data,
                backgroundColor: [
                    '#1454FE',
                    '#5B6BFF',
                    '#8B95FF',
                    '#CADBFF',
                    '#E8EBFF'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        padding: 20,
                        font: {
                            size: 14,
                            family: 'Pretendard'
                        },
                        generateLabels: function(chart) {
                            const data = chart.data;
                            if (data.labels.length && data.datasets.length) {
                                return data.labels.map((label, i) => {
                                    const dataset = data.datasets[0];
                                    const value = dataset.data[i];
                                    return {
                                        text: `${label} ${value}%`,
                                        fillStyle: dataset.backgroundColor[i],
                                        hidden: false,
                                        index: i
                                    };
                                });
                            }
                            return [];
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            return `${label}: ${value}%`;
                        }
                    }
                }
            },
            onClick: handleChartClick
        }
    };

    // 차트 생성
    const chart = new Chart(chartCanvas, config);

    // 차트 클릭 이벤트 핸들러
    function handleChartClick(event, activeElements) {
        if (activeElements.length > 0) {
            const index = activeElements[0].index;
            const category = portfolioData.labels[index];
            const stocks = portfolioData.stocks[category] || window.portfolioData?.stocks?.[category];
            
            if (stocks && stocks.length > 0) {
                showStockModal(category, stocks);
            }
        }
    }

    // 종목 상세 모달 표시
    window.showStockModal = function showStockModal(category, stocks) {
        // 기존 모달 제거
        const existingModal = document.getElementById('stockDetailModal');
        if (existingModal) {
            existingModal.remove();
        }

        // 모달 HTML 생성
        const modalHTML = `
            <div id="stockDetailModal" class="modal fade" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${category} 상세 종목</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="stock-list">
                                ${stocks.map(stock => `
                                    <div class="stock-item">
                                        <div class="stock-info">
                                            <h6 class="stock-name">${stock.name}</h6>
                                            <span class="stock-code">${stock.code}</span>
                                        </div>
                                        <div class="stock-details">
                                            <div class="allocation">
                                                <span class="label">비중</span>
                                                <span class="value">${stock.allocation}%</span>
                                            </div>
                                            ${stock.currentPrice ? `
                                                <div class="price">
                                                    <span class="label">현재가</span>
                                                    <span class="value">${formatNumber(stock.currentPrice)}원</span>
                                                </div>
                                            ` : ''}
                                            ${stock.change ? `
                                                <div class="change ${stock.change > 0 ? 'up' : 'down'}">
                                                    <span class="label">변동률</span>
                                                    <span class="value">${stock.change > 0 ? '+' : ''}${stock.change}%</span>
                                                </div>
                                            ` : ''}
                                        </div>
                                        <button class="btn btn-sm btn-outline-primary view-detail-btn" 
                                                onclick="viewStockDetail('${stock.code || stock.ticker}')">
                                            자세히 보기
                                        </button>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // 모달 추가 및 표시
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        const modal = new bootstrap.Modal(document.getElementById('stockDetailModal'));
        modal.show();
    }

    // 숫자 포맷팅
    function formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }
});

// 종목 상세 페이지로 이동
function viewStockDetail(stockCode) {
    // 상세 보고서 모달 표시
    showStockReportModal(stockCode);
}

// 종목 상세 보고서 모달 표시
window.showStockReportModal = async function showStockReportModal(ticker) {
    try {
        // API 호출하여 종목 상세 정보 가져오기
        const response = await fetch(`/api/stock/${ticker}/report`);
        const data = await response.json();
        
        if (!data.success) {
            alert('종목 정보를 불러올 수 없습니다.');
            return;
        }
        
        const report = data.report;
        
        // 기존 모달 제거
        const existingModal = document.getElementById('stockReportModal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // 보고서 모달 HTML 생성
        const modalHTML = `
            <div id="stockReportModal" class="modal fade" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${report.name} (${report.ticker}) 상세 분석 보고서</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="report-section">
                                <h6 class="section-title">기본 정보</h6>
                                <div class="info-grid">
                                    <div class="info-item">
                                        <span class="label">종목명</span>
                                        <span class="value">${report.name}</span>
                                    </div>
                                    <div class="info-item">
                                        <span class="label">종목코드</span>
                                        <span class="value">${report.ticker}</span>
                                    </div>
                                    <div class="info-item">
                                        <span class="label">현재가</span>
                                        <span class="value">${formatNumber(report.currentPrice)}원</span>
                                    </div>
                                    <div class="info-item">
                                        <span class="label">시가총액</span>
                                        <span class="value">${formatMarketCap(report.marketCap)}</span>
                                    </div>
                                    <div class="info-item">
                                        <span class="label">평가점수</span>
                                        <span class="value score-${getScoreClass(report.score)}">${report.score}점</span>
                                    </div>
                                    <div class="info-item">
                                        <span class="label">종합평가</span>
                                        <span class="value evaluation-${report.evaluation}">${report.evaluation}</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="report-section">
                                <h6 class="section-title">재무 지표</h6>
                                <div class="metrics-grid">
                                    <div class="metric-item">
                                        <div class="metric-label">매출성장률</div>
                                        <div class="metric-value ${getMetricClass('growth', report.metrics.revenueGrowth)}">
                                            ${report.metrics.revenueGrowth !== null ? report.metrics.revenueGrowth.toFixed(1) + '%' : '-'}
                                        </div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-label">순이익률</div>
                                        <div class="metric-value ${getMetricClass('profit', report.metrics.profitMargin)}">
                                            ${report.metrics.profitMargin !== null ? report.metrics.profitMargin.toFixed(1) + '%' : '-'}
                                        </div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-label">부채비율</div>
                                        <div class="metric-value ${getMetricClass('debt', report.metrics.debtRatio)}">
                                            ${report.metrics.debtRatio !== null ? report.metrics.debtRatio.toFixed(1) + '%' : '-'}
                                        </div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-label">PER</div>
                                        <div class="metric-value ${getMetricClass('per', report.metrics.per)}">
                                            ${report.metrics.per !== null ? report.metrics.per.toFixed(1) : '-'}
                                        </div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-label">PBR</div>
                                        <div class="metric-value ${getMetricClass('pbr', report.metrics.pbr)}">
                                            ${report.metrics.pbr !== null ? report.metrics.pbr.toFixed(2) : '-'}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="report-section">
                                <h6 class="section-title">평가 분석</h6>
                                <div class="evaluation-reasons">
                                    ${report.evaluationReasons.map(reason => 
                                        `<div class="reason-item">
                                            <i class="fas fa-check-circle"></i>
                                            <span>${reason}</span>
                                        </div>`
                                    ).join('')}
                                </div>
                            </div>
                            
                            <div class="report-footer">
                                <small class="text-muted">기준일: ${report.updatedAt}</small>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">닫기</button>
                            <button type="button" class="btn btn-primary" onclick="window.location.href='/stock?ticker=${report.ticker}'">
                                상세 페이지로 이동
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 모달 추가 및 표시
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        const modal = new bootstrap.Modal(document.getElementById('stockReportModal'));
        modal.show();
        
    } catch (error) {
        console.error('종목 보고서 로드 실패:', error);
        alert('종목 정보를 불러오는 중 오류가 발생했습니다.');
    }
}

// 헬퍼 함수들
function formatMarketCap(value) {
    if (!value) return '-';
    if (value >= 1000000000000) {
        return (value / 1000000000000).toFixed(1) + '조원';
    } else if (value >= 100000000) {
        return (value / 100000000).toFixed(0) + '억원';
    } else {
        return formatNumber(value) + '원';
    }
}

function getScoreClass(score) {
    if (score >= 80) return 'excellent';
    if (score >= 70) return 'good';
    if (score >= 60) return 'fair';
    return 'poor';
}

function getMetricClass(type, value) {
    if (value === null || value === undefined) return '';
    
    switch(type) {
        case 'growth':
            return value > 20 ? 'excellent' : value > 10 ? 'good' : value > 0 ? 'fair' : 'poor';
        case 'profit':
            return value > 15 ? 'excellent' : value > 10 ? 'good' : value > 5 ? 'fair' : 'poor';
        case 'debt':
            return value < 50 ? 'excellent' : value < 100 ? 'good' : value < 150 ? 'fair' : 'poor';
        case 'per':
            return value < 10 ? 'excellent' : value < 15 ? 'good' : value < 25 ? 'fair' : 'poor';
        case 'pbr':
            return value < 1 ? 'excellent' : value < 1.5 ? 'good' : value < 3 ? 'fair' : 'poor';
        default:
            return '';
    }
}

// 스타일 추가
const style = document.createElement('style');
style.textContent = `
    .stock-list {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    .stock-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px;
        background: #f8f9fa;
        border-radius: 12px;
        transition: all 0.2s ease;
    }

    .stock-item:hover {
        background: #e9ecef;
        transform: translateY(-1px);
    }

    .stock-info {
        flex: 1;
    }

    .stock-name {
        font-size: 16px;
        font-weight: 600;
        margin: 0 0 4px 0;
        color: #000;
    }

    .stock-code {
        font-size: 14px;
        color: #6c757d;
    }

    .stock-details {
        display: flex;
        gap: 24px;
        margin: 0 24px;
    }

    .stock-details > div {
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .stock-details .label {
        font-size: 12px;
        color: #6c757d;
        margin-bottom: 4px;
    }

    .stock-details .value {
        font-size: 16px;
        font-weight: 600;
        color: #000;
    }

    .stock-details .change.up .value {
        color: #FE3F14;
    }

    .stock-details .change.down .value {
        color: #1454FE;
    }

    .view-detail-btn {
        flex-shrink: 0;
    }

    /* 보고서 모달 스타일 */
    .report-section {
        margin-bottom: 24px;
        padding-bottom: 24px;
        border-bottom: 1px solid #e9ecef;
    }

    .report-section:last-of-type {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #000;
        margin-bottom: 16px;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
    }

    .info-item {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .info-item .label {
        font-size: 14px;
        color: #6c757d;
    }

    .info-item .value {
        font-size: 16px;
        font-weight: 600;
        color: #000;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
    }

    .metric-item {
        text-align: center;
        padding: 16px;
        background: #f8f9fa;
        border-radius: 12px;
    }

    .metric-label {
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
    }

    .metric-value.excellent {
        color: #198754;
    }

    .metric-value.good {
        color: #0dcaf0;
    }

    .metric-value.fair {
        color: #ffc107;
    }

    .metric-value.poor {
        color: #dc3545;
    }

    .score-excellent {
        color: #198754;
    }

    .score-good {
        color: #0dcaf0;
    }

    .score-fair {
        color: #ffc107;
    }

    .score-poor {
        color: #dc3545;
    }

    .evaluation-매수 {
        color: #198754;
        font-weight: 700;
    }

    .evaluation-보유 {
        color: #0dcaf0;
        font-weight: 700;
    }

    .evaluation-매도 {
        color: #dc3545;
        font-weight: 700;
    }

    .evaluation-reasons {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .reason-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        background: #f8f9fa;
        border-radius: 8px;
    }

    .reason-item i {
        color: #198754;
        font-size: 16px;
    }

    .report-footer {
        margin-top: 24px;
        text-align: center;
    }

    @media (max-width: 768px) {
        .stock-item {
            flex-direction: column;
            gap: 12px;
            text-align: center;
        }

        .stock-details {
            margin: 0;
        }

        .info-grid,
        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 576px) {
        .info-grid,
        .metrics-grid {
            grid-template-columns: 1fr;
        }
    }
`;
document.head.appendChild(style);