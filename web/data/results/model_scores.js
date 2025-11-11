// 模型分数配置
// 这个文件包含各个模型的评估分数
// 可以根据实际情况修改这些数据

const modelScores = {
    "model-a": {
        "overall": 2.50,
        "metric1": 2.30,
        "metric2": 2.40,
        "metric3": 2.60,
        "metric4": 2.70,
        "category1": 2.50,
        "category2": 2.40,
        "category3": 2.60
    },
    "model-b": {
        "overall": 2.20,
        "metric1": 2.10,
        "metric2": 2.30,
        "metric3": 2.25,
        "metric4": 2.15,
        "category1": 2.30,
        "category2": 2.25,
        "category3": 2.35
    },
    "model-c": {
        "overall": 1.90,
        "metric1": 1.80,
        "metric2": 2.00,
        "metric3": 1.95,
        "metric4": 1.85,
        "category1": 2.10,
        "category2": 2.00,
        "category3": 2.05
    }
};

// 如果需要在其他地方使用，可以导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = modelScores;
}
