# 配置文件说明

## 文件结构

```
WEB/
├── data/
│   └── results/
│       ├── output_folders.js    # 输出文件夹配置
│       └── model_scores.js      # 模型分数配置
├── visualizer/
│   └── data/
│       └── data_public.js       # 公共数据配置
├── static/
│   ├── css/
│   │   ├── index.css            # 自定义样式
│   │   └── leaderboard.css      # 排行榜样式
│   └── js/
│       ├── sort-table.js        # 表格排序功能
│       ├── explorer-index.js    # 探索器功能
│       ├── question_card.js     # 问题卡片功能
│       └── leaderboard_testmini.js  # 排行榜测试功能
└── template.html                # 主 HTML 文件
```

## 必需的第三方资源

以下资源需要从 CDN 或本地文件加载：

### CSS 框架
- Bulma CSS (bulma.min.css)
- Bulma Carousel (bulma-carousel.min.css)
- Bulma Slider (bulma-slider.min.css)
- Font Awesome (fontawesome.all.min.css)
- Academicons (从 CDN 加载)

### JavaScript 库
- jQuery (从 CDN 加载)
- Bulma Carousel (bulma-carousel.min.js)
- Bulma Slider (bulma-slider.min.js)
- Font Awesome (fontawesome.all.min.js)

## 配置文件说明

### 1. output_folders.js
定义各个模型的结果文件夹路径。如果不需要动态加载文件夹，可以留空或删除引用。

### 2. model_scores.js
定义各个模型的评估分数。如果表格数据已经在 HTML 中硬编码，可以留空或删除引用。

### 3. data_public.js
包含可视化相关的公共数据。如果不需要可视化功能，可以留空或删除引用。

## 如何获取缺失的资源

### 方法 1: 使用 CDN（推荐）

在 `template.html` 中，以下资源已经从 CDN 加载：
- jQuery: `https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js`
- Academicons: `https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css`

### 方法 2: 下载到本地

1. **Bulma CSS**: 从 [Bulma 官网](https://bulma.io/) 下载
2. **Bulma Carousel**: 从 [GitHub](https://github.com/Wikiki/bulma-carousel) 下载
3. **Bulma Slider**: 从 [GitHub](https://github.com/Wikiki/bulma-slider) 下载
4. **Font Awesome**: 从 [Font Awesome 官网](https://fontawesome.com/) 下载

### 方法 3: 使用 npm/yarn

```bash
npm install bulma bulma-carousel bulma-slider @fortawesome/fontawesome-free
```

然后将文件复制到 `static/` 目录。

## 图片资源

将以下图片放入 `static/images/` 目录：

- `intro.png` - 介绍图片
- `radar.png` - 雷达图
- `category.png` - 类别分布图
- `wordcloud.png` - 词云图
- `institution1.png`, `institution2.png`, `institution3.png` - 机构 Logo
- 结果展示的 GIF 或图片文件

## 故障排除

### 问题 1: 页面没有样式
- 检查 `static/css/` 目录下是否有 CSS 文件
- 检查 HTML 中的 CSS 文件路径是否正确
- 检查浏览器控制台是否有 404 错误

### 问题 2: JavaScript 功能不工作
- 检查 `static/js/` 目录下是否有 JS 文件
- 检查 HTML 中的 JS 文件路径是否正确
- 检查浏览器控制台是否有 JavaScript 错误
- 确保 jQuery 已加载（其他脚本依赖 jQuery）

### 问题 3: 图片不显示
- 检查图片路径是否正确
- 检查图片文件是否存在
- 检查浏览器控制台是否有 404 错误

### 问题 4: 表格排序不工作
- 确保 `sort-table.js` 已加载
- 检查表格是否有 `js-sort-table` 类
- 检查浏览器控制台是否有 JavaScript 错误

## 简化版本

如果不需要某些功能，可以：

1. **删除不需要的 JS 文件引用**：从 HTML 中删除对应的 `<script>` 标签
2. **删除不需要的 CSS 文件引用**：从 HTML 中删除对应的 `<link>` 标签
3. **简化数据文件**：如果数据已经在 HTML 中硬编码，可以删除或简化数据文件

## 快速开始

1. 确保所有目录结构已创建
2. 下载或配置必需的 CSS 和 JS 文件
3. 添加图片资源
4. 根据实际情况修改配置文件
5. 在浏览器中打开 `template.html` 查看效果
