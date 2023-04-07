/*
* @desc Knn算法
* @param  {Object} current 
* @param  {Array} points 
* @param  {Number} k 
* @param  {Function} c        
* @return {Array}
*/
function getKnn(current, points, labelX, labelY, k, c) {
    var dists = [];//存放最接近的
    var classify = [];//分类标识
    points.map(function (item) {
        if (classify.indexOf(item[labelY]) < 0) classify.push(item[labelY]);
        var result = {};
        result.p = item; 
        result.d = c(current, item[labelX]) ;
        dists.push(result);
    });
    dists.sort(function (a, b) {//排序
        return a.d - b.d;
    });
    return { dists: dists.slice(0, k), classify: classify };
}

/*
* @desc 决策
* @param  {Object} current 输入值
* @param  {Object} points 训练样本集
* @param  {Object} labelX 用于分类的输入值
* @param  {Object} labelY 用于分类的输出值
* @param  {Number} k 用于选择最近邻的数目
* @param  {Function} c 自定义比较函数
* @return {Object} 
*/
function classify0(current, points, labelX, labelY, k, c) {
    var result = [];
    var knn = getKnn(current, points, labelX, labelY, k, c);
    var dists = knn.dists;
    for (var i of knn.classify) {
        result.push({
            label: i,
            value: 0
        });
    }
    dists.map(function (item) {
        for (var i of result) {
            if (i.label === item.p[labelY]) {
                i.value++;
                break;
            }
        }
    });
    result.sort(function (a, b) {
        return b.value - a.value;
    });
    return { result: result[0].label, dists: dists };
}