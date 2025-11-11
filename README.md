# GGBench 网站

该目录包含 GGBench 项目官网的全部静态资源，展示数据集、评测基准与最新结果。页面已完整配置，可直接部署或二次开发。

## 目录结构

```
web/
├── index.html          # 主页 HTML
├── README.md              # 使用说明
├── CONFIG.md              # 额外配置说明
├── static/
│   ├── css/
│   │   ├── index.css      # 全局样式
│   │   └── leaderboard.css# 排行榜粘性表头与样式
│   ├── images/            # 页面图片资源
│   └── js/
│       ├── sort-table.js  # 表格排序
│       ├── explorer-index.js
│       ├── question_card.js
│       └── leaderboard_testmini.js
├── data/
│   └── results/
│       ├── output_folders.js
│       └── model_scores.js
└── visualizer/
    └── data/
        └── data_public.js
```

## 页面概览

- **Hero 区块**：展示标题、作者、项目链接（论文、代码、数据集、Benchmark 锚点）。
- **TL;DR / Introduction**：简要介绍与图示。
- **Benchmark 概览**：雷达图、类别分布图，文字居中对齐。
- **Leaderboard**：两层粘性表头，展示模型在 Planning、Middle Process、Final Result、Overall Scores 的各项指标。
- **BibTeX 引用**：一键复制引用信息。

## 快速启动

1. **安装依赖**  
   页面引用 CDN 版本的 Bulma、Font Awesome、jQuery 与 Bulma Carousel，无需额外安装。若需离线部署，可将资源下载到本地并修改 `<link>` / `<script>`。

2. **本地预览**  
   使用任意静态服务器（如 `python -m http.server`）在 `web/` 目录下启动即可浏览。

3. **数据与图片更新**
   - 模型榜单在 `template.html` 中直接维护。
   - 样例图片位于 `static/images/`，可替换为最新分析图。
   - 如需调整结果展示卡片/交互，可修改 `static/js/` 和 `static/css/` 中文件。

4. **链接配置**  
   Hero 区的按钮已链接至：
   - 论文：`https://openreview.net/forum?id=y68PHsVGYp`
   - 代码：`https://github.com/opendatalab-raiser/GGBench`
   - 数据集：`https://huggingface.co/datasets/opendatalab-raiser/GGBench`
   若有更新可直接修改 `template.html`。


## 版权信息

页面遵循 Creative Commons Attribution-ShareAlike 4.0 International License，与项目主仓库一致。*** End Patch
