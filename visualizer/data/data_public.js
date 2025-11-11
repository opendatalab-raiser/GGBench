// 公共数据配置
// 这个文件包含可视化相关的公共数据
// 可以根据实际情况修改这些数据

const publicData = {
    categories: [
        "Category 1",
        "Category 2",
        "Category 3"
    ],
    models: [
        "Model A",
        "Model B",
        "Model C"
    ],
    metrics: [
        "Metric 1",
        "Metric 2",
        "Metric 3",
        "Metric 4"
    ]
};

// 如果需要在其他地方使用，可以导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = publicData;
}
