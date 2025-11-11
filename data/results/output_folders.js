// 输出文件夹配置
// 这个文件定义了各个模型的结果文件夹路径
// 可以根据实际情况修改这些路径

const outputFolders = {
    "model-a": "./data/results/model-a",
    "model-b": "./data/results/model-b",
    "model-c": "./data/results/model-c"
};

// 如果需要在其他地方使用，可以导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = outputFolders;
}
