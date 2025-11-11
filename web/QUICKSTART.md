# 快速启动指南

## 问题：网站无法渲染

如果网站无法正常显示，通常是因为缺少必需的 CSS 和 JS 文件。按照以下步骤解决：

## 解决方案

### 方案 1: 使用 CDN（最简单，推荐）

修改 `template.html`，将所有本地资源改为 CDN 链接：

1. **Bulma CSS** - 在 `<head>` 中添加：
```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">
```

2. **Bulma Carousel** - 替换现有链接：
```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma-carousel@4.0.4/dist/css/bulma-carousel.min.css">
<script src="https://cdn.jsdelivr.net/npm/bulma-carousel@4.0.4/dist/js/bulma-carousel.min.js"></script>
```

3. **Font Awesome** - 替换现有链接：
```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js"></script>
```

### 方案 2: 下载到本地

1. 下载 Bulma CSS：
```bash
curl -o static/css/bulma.min.css https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css
```

2. 下载 Bulma Carousel：
```bash
# CSS
curl -o static/css/bulma-carousel.min.css https://cdn.jsdelivr.net/npm/bulma-carousel@4.0.4/dist/css/bulma-carousel.min.css
# JS
curl -o static/js/bulma-carousel.min.js https://cdn.jsdelivr.net/npm/bulma-carousel@4.0.4/dist/js/bulma-carousel.min.js
```

3. 下载 Font Awesome（需要注册账号，或使用 CDN）

### 方案 3: 创建最小化版本

如果不需要某些功能，可以创建一个简化版本：

1. **移除不需要的功能**：
   - 如果不需要轮播，移除 `bulma-carousel` 相关代码
   - 如果不需要滑块，移除 `bulma-slider` 相关代码

2. **使用内联样式**：
   - 将关键的 CSS 直接写入 HTML 的 `<style>` 标签
   - 这样可以减少外部依赖

## 检查清单

在浏览器中打开网站前，检查：

- [ ] `static/css/index.css` 存在
- [ ] `static/css/leaderboard.css` 存在
- [ ] `static/js/sort-table.js` 存在
- [ ] Bulma CSS 已加载（本地或 CDN）
- [ ] jQuery 已加载（已从 CDN 加载）
- [ ] 图片文件路径正确
- [ ] 浏览器控制台没有错误

## 测试步骤

1. 打开浏览器开发者工具（F12）
2. 查看 Console 标签页，检查是否有错误
3. 查看 Network 标签页，检查哪些文件加载失败（显示红色）
4. 根据错误信息修复问题

## 常见错误

### 错误 1: "Failed to load resource"
- **原因**: 文件路径错误或文件不存在
- **解决**: 检查文件路径，确保文件存在

### 错误 2: "bulmaCarousel is not defined"
- **原因**: Bulma Carousel JS 文件未加载
- **解决**: 确保 `bulma-carousel.min.js` 已正确加载

### 错误 3: 页面没有样式
- **原因**: Bulma CSS 未加载
- **解决**: 确保 `bulma.min.css` 已正确加载

### 错误 4: 表格无法排序
- **原因**: `sort-table.js` 未加载或 jQuery 未加载
- **解决**: 确保两个文件都已加载，且 jQuery 在 `sort-table.js` 之前加载

## 最小可用版本

如果以上方法都不行，可以创建一个最小可用版本：

1. 移除所有外部 JS 库依赖
2. 使用纯 CSS 实现样式
3. 使用原生 JavaScript 实现交互
4. 逐步添加功能

## 获取帮助

如果遇到问题：
1. 检查浏览器控制台的错误信息
2. 查看 `CONFIG.md` 了解详细配置
3. 参考原始 `ui.html` 文件的结构
