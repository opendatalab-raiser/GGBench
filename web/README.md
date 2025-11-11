# 网站使用说明

## 文件结构

```
WEB/
├── template.html        # 新的模板文件
├── README.md            # 本说明文件
├── CONFIG.md            # 配置文件说明
├── data/
│   └── results/
│       ├── output_folders.js    # 输出文件夹配置
│       └── model_scores.js      # 模型分数配置
├── visualizer/
│   └── data/
│       └── data_public.js       # 公共数据配置
└── static/
    ├── css/
    │   ├── index.css            # 自定义样式
    │   └── leaderboard.css      # 排行榜样式
    └── js/
        ├── sort-table.js        # 表格排序功能
        ├── explorer-index.js    # 探索器功能
        ├── question_card.js     # 问题卡片功能
        └── leaderboard_testmini.js  # 排行榜测试功能
```

## 主要功能

1. **响应式设计** - 使用 Bulma CSS 框架
2. **导航栏** - 包含研究项目链接
3. **Hero Section** - 标题、作者、机构、链接按钮
4. **内容展示** - TL;DR、介绍、分析等章节
5. **结果展示** - 分类标签、卡片式布局、可折叠详情
6. **排行榜** - 可排序的表格
7. **BibTeX** - 一键复制功能
8. **机构标识** - 底部机构 Logo

## 使用步骤

### 1. 自定义内容

编辑 `template.html`，替换以下占位符：

- **标题和副标题**：搜索 "Your Research Project Title"
- **作者信息**：修改作者姓名、链接、机构
- **TL;DR 和介绍**：替换为你的研究内容
- **类别和结果**：添加你的展示类别和案例
- **排行榜数据**：更新模型和分数
- **BibTeX**：修改引用信息

### 2. 添加图片资源

将你的图片放入 `static/images/` 目录：

- `intro.png` - 介绍图片
- `radar.png` - 雷达图
- `category.png` - 类别分布图
- `wordcloud.png` - 词云图
- `institution1.png`, `institution2.png`, `institution3.png` - 机构 Logo
- 结果展示的 GIF 或图片

### 3. 配置类别和案例

在 "Results Showcase" 部分：

```html
<div class="box m-5">
    <div class="content has-text-centered">
        <h3 class="subtitle is-5">Category Name</h3>
        <!-- 添加你的案例 -->
        <div class="column">
            <img class="video-gif"
                src="static/images/your-image.gif"
                alt="case 1" 
                data-question="Your question"
                data-prompt="Your prompt" />
        </div>
    </div>
</div>
```

### 4. 更新排行榜

修改两个表格：
- `results-model-dimension` - 模型维度性能
- `results-per-category` - 类别性能

### 5. 链接配置

更新以下链接：
- Paper PDF 链接
- Code GitHub 链接
- Dataset 链接
- 作者个人主页链接
- 机构网站链接

## 依赖文件

### 必需的第三方库（需要下载或使用 CDN）

以下文件需要从相应来源获取：

1. **Bulma CSS 框架**
   - `bulma.min.css` - 从 [Bulma 官网](https://bulma.io/) 下载
   - `bulma-carousel.min.css` 和 `bulma-carousel.min.js` - 从 [GitHub](https://github.com/Wikiki/bulma-carousel) 下载
   - `bulma-slider.min.css` 和 `bulma-slider.min.js` - 从 [GitHub](https://github.com/Wikiki/bulma-slider) 下载

2. **Font Awesome**
   - `fontawesome.all.min.css` 和 `fontawesome.all.min.js` - 从 [Font Awesome](https://fontawesome.com/) 下载

3. **jQuery** (已从 CDN 加载，无需下载)

### 已创建的文件

以下文件已经创建，无需额外配置：

```
static/
├── css/
│   ├── index.css            # ✅ 已创建
│   └── leaderboard.css      # ✅ 已创建
├── js/
│   ├── sort-table.js        # ✅ 已创建
│   ├── explorer-index.js    # ✅ 已创建
│   ├── question_card.js     # ✅ 已创建
│   └── leaderboard_testmini.js  # ✅ 已创建
data/
└── results/
    ├── output_folders.js    # ✅ 已创建
    └── model_scores.js      # ✅ 已创建
visualizer/
└── data/
    └── data_public.js       # ✅ 已创建
```

### 需要添加的图片

将以下图片放入 `static/images/` 目录：

- `intro.png` - 介绍图片
- `radar.png` - 雷达图
- `category.png` - 类别分布图
- `wordcloud.png` - 词云图
- `institution1.png`, `institution2.png`, `institution3.png` - 机构 Logo
- 结果展示的 GIF 或图片文件

## 自定义样式

所有自定义样式都在 `<style>` 标签中，主要类名：

- `.rv-*` - 结果展示相关样式
- `.rv-filterbar` - 过滤栏
- `.rv-tabs` - 标签按钮
- `.rv-card` - 卡片样式
- `.rv-details` - 可折叠详情

可以根据需要修改颜色、间距、布局等。

## 浏览器兼容性

- Chrome/Edge (推荐)
- Firefox
- Safari
- 移动端浏览器

## 注意事项

1. 确保所有外部资源链接正确
2. JavaScript 功能需要 jQuery 和 Bulma Carousel
3. 图片路径使用相对路径
4. 表格排序功能需要 `sort-table.js`
5. 复制 BibTeX 功能需要浏览器支持 Clipboard API

## 许可证

与原项目相同，使用 Creative Commons Attribution-ShareAlike 4.0 International License。
