// 表格排序功能
// 为带有 js-sort-table 类的表格添加排序功能

(function() {
    'use strict';

    // 初始化所有可排序表格
    function initSortableTables() {
        const tables = document.querySelectorAll('.js-sort-table');
        tables.forEach(table => {
            makeSortable(table);
        });
    }

    // 使表格可排序
    function makeSortable(table) {
        const headers = table.querySelectorAll('thead th');
        headers.forEach((header, index) => {
            // 跳过第一列（通常是序号）
            if (index === 0) return;
            
            header.style.cursor = 'pointer';
            header.setAttribute('data-sort', 'none');
            header.addEventListener('click', () => {
                sortTable(table, index, header);
            });
        });
    }

    // 排序表格
    function sortTable(table, columnIndex, header) {
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const currentSort = header.getAttribute('data-sort');
        
        // 重置所有标题的排序状态
        const headers = table.querySelectorAll('thead th');
        headers.forEach(h => {
            if (h !== header) {
                h.setAttribute('data-sort', 'none');
                h.classList.remove('sort-asc', 'sort-desc');
            }
        });

        // 切换排序方向
        let newSort = 'asc';
        if (currentSort === 'asc') {
            newSort = 'desc';
        }

        // 排序行
        rows.sort((a, b) => {
            const aText = a.cells[columnIndex].textContent.trim();
            const bText = b.cells[columnIndex].textContent.trim();
            
            // 尝试解析为数字
            const aNum = parseFloat(aText);
            const bNum = parseFloat(bText);
            
            if (!isNaN(aNum) && !isNaN(bNum)) {
                return newSort === 'asc' ? aNum - bNum : bNum - aNum;
            }
            
            // 字符串比较
            return newSort === 'asc' 
                ? aText.localeCompare(bText)
                : bText.localeCompare(aText);
        });

        // 重新插入排序后的行
        rows.forEach(row => tbody.appendChild(row));

        // 更新标题状态
        header.setAttribute('data-sort', newSort);
        header.classList.remove('sort-asc', 'sort-desc');
        header.classList.add(`sort-${newSort}`);
    }

    // 当 DOM 加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSortableTables);
    } else {
        initSortableTables();
    }
})();
