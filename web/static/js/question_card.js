// 问题卡片功能
// 处理问题卡片的显示和交互

(function() {
    'use strict';
    
    // 初始化问题卡片
    function initQuestionCards() {
        const cards = document.querySelectorAll('.question-card');
        cards.forEach(card => {
            // 可以在这里添加卡片相关的交互功能
        });
    }
    
    // 当 DOM 加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initQuestionCards);
    } else {
        initQuestionCards();
    }
})();
