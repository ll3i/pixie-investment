/**
 * 투자 분석 웹 애플리케이션 메인 자바스크립트
 */

// DOM이 로드된 후 실행
document.addEventListener('DOMContentLoaded', function() {
    // 부트스트랩 툴팁 초기화
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // 네비게이션 바의 현재 페이지 표시
    highlightCurrentPage();
    
    // 페이드 인 효과 추가
    addFadeInEffects();
});

/**
 * 현재 페이지에 해당하는 네비게이션 메뉴 강조
 */
function highlightCurrentPage() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        
        if (href === currentPath || 
            (currentPath.startsWith(href) && href !== '/')) {
            link.classList.add('active');
        }
    });
}

/**
 * 페이지 요소에 페이드 인 효과 추가
 */
function addFadeInEffects() {
    const elements = document.querySelectorAll('.card, .result-section');
    
    elements.forEach((element, index) => {
        element.classList.add('fade-in');
        element.style.animationDelay = `${index * 0.15}s`;
    });
}

/**
 * 숫자 포맷팅 함수
 * @param {number} num - 포맷팅할 숫자
 * @param {boolean} hasCurrency - 통화 표시 여부
 * @returns {string} 포맷팅된 문자열
 */
function formatNumber(num, hasCurrency = false) {
    if (isNaN(num)) return '0';
    
    const formatted = new Intl.NumberFormat('ko-KR').format(num);
    return hasCurrency ? formatted + '원' : formatted;
}

/**
 * 백엔드 API 호출 유틸리티 함수
 * @param {string} url - API 엔드포인트
 * @param {object} options - fetch 옵션
 * @returns {Promise} API 응답 프로미스
 */
async function callApi(url, options = {}) {
    try {
        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`API 호출 실패: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API 호출 중 오류 발생:', error);
        throw error;
    }
}

/**
 * 날짜 포맷팅 함수
 * @param {Date|string} date - 포맷팅할 날짜
 * @returns {string} YYYY-MM-DD 형식의 날짜 문자열
 */
function formatDate(date) {
    const d = new Date(date);
    
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    
    return `${year}-${month}-${day}`;
}

/**
 * 스크롤을 부드럽게 특정 요소로 이동
 * @param {string} elementId - 스크롤할 요소의 ID
 */
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
} 