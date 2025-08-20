import pandas as pd
import os
import re

def extract_tf_name(file_path):
    """从文件路径中提取转录因子名称"""
    filename = os.path.basename(file_path)
    # 使用正则表达式匹配文件名".bed"之前的字母
    match = re.search(r'([^\/]+)\.bed', filename)
    if match:
        return match.group(1)  # 返回匹配的第一个组的内容，即".bed"之前的字母
    else:
        return None  # 如果没有匹配到，返回None

def load_bed_file(file_path, tf_name):
    """加载BED文件并返回DataFrame，同时添加转录因子名称"""
    df = pd.read_csv(file_path, sep='\t', header=None)
    df['tf_name'] = tf_name  # 添加转录因子名称列
    return df

def find_overlaps(tf_df, promoter_df):
    """查找转录因子与启动子区域的重叠"""
    overlaps = []
    
    for _, tf_row in tf_df.iterrows():
        tf_chr, tf_start, tf_end, tf_name = tf_row[0], tf_row[1], tf_row[2], tf_row['tf_name']
        
        # 查找重叠的启动子区域
        promoter_overlap = promoter_df[
            (promoter_df[0] == tf_chr) &
            (promoter_df[1] < tf_end) &
            (promoter_df[2] > tf_start)
        ]
        
        for _, promoter_row in promoter_overlap.iterrows():
            promoter_start = promoter_row[1]
            promoter_end = promoter_row[2]
           
            
            # 计算重叠区域
            overlap_start = max(tf_start, promoter_start)
            overlap_end = min(tf_end, promoter_end)
            overlap_length = max(0, overlap_end - overlap_start)  # 确保重叠长度不为负
            
            overlaps.append({
                'tf_chr': tf_chr,
                'tf_start': tf_start,
                'tf_end': tf_end,
                'promoter_start': promoter_start,
                'promoter_end': promoter_end,
                'gene_name': promoter_row[4],
                'strand': promoter_row[3],
                'tf_name': tf_name,  # 添加转录因子名称
                'TPM-GM12878': promoter_row[5],  # 添加TPM-GM12878表达值
                'overlap_start': overlap_start,  # 添加重叠起始位置
                'overlap_end': overlap_end,      # 添加重叠结束位置
                'overlap_length': overlap_length   # 添加重叠长度
            })
    
    return pd.DataFrame(overlaps)

def sort_overlaps(overlap_df):
    """根据转录因子的起始位点排序"""
    return overlap_df.sort_values(by=['tf_start'])

def main():
    # 填入所有转录因子的BED文件路径
    tf_files = [
        '/TF sequence/storm.bed(GM12878)/SP1.bed',
        '/TF sequence/storm.bed(GM12878)/ETS1.bed',
        '/TF sequence/storm.bed(GM12878)/NFYB.bed',
        '/TF sequence/storm.bed(GM12878)/TBP.bed',
        '/TF sequence/storm.bed(GM12878)/BHLHE40.bed',
        '/TF sequence/storm.bed(GM12878)/NRF1.bed',
        '/TF sequence/storm.bed(GM12878)/YY1.bed',
        '/TF sequence/storm.bed(GM12878)/POLAR2A.bed',
        '/TF sequence/storm.bed(GM12878)/CTCF.bed',
        '/TF sequence/storm.bed(GM12878)/YBX1.bed',
        '/TF sequence/storm.bed(GM12878)/CEBPZ.bed',
        '/TF sequence/storm.bed(GM12878)/NFYA.bed',
        '/TF sequence/storm.bed(GM12878)/NR2C2.bed'


        #   '/TF sequence/storm.bed(K562)/TBP.bed',
        #   '/TF sequence/storm.bed(K562)/BHLHE40.bed',
        #   '/TF sequence/storm.bed(K562)/ETS1.bed',
        #   '/TF sequence/storm.bed(K562)/NRF1.bed',
        #   '/TF sequence/storm.bed(K562)/SP1.bed',
        #   '/TF sequence/storm.bed(K562)/YY1.bed',
        #   '/TF sequence/storm.bed(K562)/NFYB.bed',
        #   '/TF sequence/storm.bed(K562)/POLAR2A.bed',
        #   '/TF sequence/storm.bed(K562)/CTCF.bed',
        #   '/TF sequence/storm.bed(K562)/YBX1.bed',
        #   '/TF sequence/storm.bed(K562)/CEBPZ.bed',
        #   '/TF sequence/storm.bed(K562)/NFYA.bed',
        #   '/TF sequence/storm.bed(K562)/NR2C2.bed'
    ]
    
    promoter_file = '/TF sequence/promoters.bed'
    
    # 加载启动子区域的BED数据
    promoter_df = load_bed_file(promoter_file, 'promoters')
    
    # 合并所有转录因子的BED数据
    tf_dataframes = [load_bed_file(tf_file, extract_tf_name(tf_file)) for tf_file in tf_files]
    tf_df = pd.concat(tf_dataframes, ignore_index=True)

    # 找到重叠
    overlaps = find_overlaps(tf_df, promoter_df)

    # 排序重叠结果
    sorted_overlaps = sort_overlaps(overlaps)

    # 输出结果
    sorted_overlaps.to_csv('/TF sequence/overlaps(GM12878).csv', index=False, sep='\t')

if __name__ == "__main__":
    main()
