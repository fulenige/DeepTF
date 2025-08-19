# 安装并加载必要的包
if (!requireNamespace("ggseqlogo", quietly = TRUE))
    install.packages("ggseqlogo")
library(ggseqlogo)
library(ggplot2)  

# 创建示例数据（替换为你自己的数据:letter_frequenc--PFM）
# 确保矩阵的列数与dimnames的列名数量一致
pfm_data <- matrix(c(
58,53,76,6,6,15,116,81,
32,50,45,139,45,110,31,18,
52,62,36,36,14,16,67,73,
37,36,40,76,27,34,28,28,
32,37,53,3,14,22,33,34,
16,24,26,7,56,18,31,7,
72,62,66,6,37,21,45,57,
69,64,99,32,16,132,71,40,
25,45,53,230,227,9,20,24,
0,2,0,0,0,0,1,0,
2,7,9,0,9,10,4,1,
12,15,12,15,34,39,7,12,
2,1,1,0,3,1,3,3,
17,22,14,14,84,113,43,77), 
    nrow = 14, byrow = TRUE,
    dimnames = list(c("T","S","Y","B","E","F","P","C","N","X","Z","A","R","O"),
                    c("Position1", "Position2", "Position3", "Position4", "Position5", "Position6", "Position7", "Position8")))
# 打印 PFM 矩阵，确认数据结构 
print("PFM 矩阵：") 
print(pfm_data)

# 转换为概率矩阵（按列标准化）
ppm_probs <- apply(pfm_data, 2, function(x) x/sum(x))
print("PPM 概率矩阵：")
print(ppm_probs)

# 使用 make_col_scheme() 创建自定义配色方案
# 定义字符和颜色
custom_col_scheme <- make_col_scheme(
    chars = c("T", "S", "Y", "B", "E", "F", "P", "C", "N", "X", "Z", "A", "R", "O"),
    cols = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#3300cc", "#bcbd22", "#17becf", "#2e8b57", "#ff6347", "#6a5acd", "#ff8c00"),
    name = "custom_col_scheme")

# 查看自定义配色方案
print("自定义配色方案：")
print(custom_col_scheme)

# 绘制logo图，应用自定义配色方案
p <- ggseqlogo(ppm_probs, 
               method = 'bits', 
               namespace = rownames(ppm_probs),
               col_scheme = custom_col_scheme) +  
    theme_classic(base_size = 14) +
    labs(y = "bits") +
    theme(
        legend.position = "none",  # 隐藏图例
        axis.text = element_text(size = 18, color = "black"),  # 设置刻度字体大小和颜色
        axis.title = element_text(size = 18, color = "black")  # 设置坐标轴标题的字体大小和颜色
    )

# 显示图形
print(p)