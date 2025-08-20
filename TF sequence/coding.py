import pandas as pd

def main():
    # 读取overlaps.csv文件
    overlaps_df = pd.read_csv('/TF sequence/overlaps(GM12878).csv', sep='\t')

    # 读取TSS.bed文件
    bed_df = pd.read_csv('/TF sequence/storm.bed(GM12878)/TSS.bed', sep='\t')

    # 转录因子映射
    tf_mapping = {
        "TBP": "T",
        "SP1": "S",
        "YY1": "Y",
        "BHLHE40": "B",
        "ETS1": "E",
        "NFYB": "F",
        "POLAR2A": "P",
        "CTCF": "C",
        "NRF1": "N",
        "YBX1": "X",
        "CEBPZ": "Z",
        "NFYA": "A",
        "NR2C2": "R",
        "a": "O"
    }

    # 用于存储TSS和中心坐标的列表
    tss_and_centers = []

    # 处理每个基因的转录因子序列
    for gene_name, group_df in overlaps_df.groupby('gene_name'):
        # 从overlaps.csv中添加事件
        for index, row in group_df.iterrows():
            tf_name = row['tf_name']
            overlap_start = row['overlap_start']
            overlap_end = row['overlap_end']
            tf_letter = tf_mapping.get(tf_name)
            if tf_letter:
                center = (overlap_start + overlap_end) // 2
                tss_and_centers.append((gene_name, center, 'overlap', tf_letter, row['strand']))

        # 从B.bed中添加事件
        gene_bed_df = bed_df[bed_df['gene'] == gene_name]
        for index, row in gene_bed_df.iterrows():
            tf_name = row['TF']
            TSS = row['TSS']
            tf_letter = tf_mapping.get(tf_name)
            if tf_letter:
                tss_and_centers.append((gene_name, TSS, 'bed', tf_letter, row['strand']))

    # 将列表转换为DataFrame
    tss_and_centers_df = pd.DataFrame(tss_and_centers, columns=['Gene', 'Position', 'Source', 'TF_Letter', 'Strand'])

    # 按照基因和位置信息排序
    tss_and_centers_df.sort_values(by=['Gene', 'Position'], ascending=[True, True], inplace=True)
     # 输出排序后的TSS和中心坐标到指定文件
    sorted_tss_and_centers_path = '/TF sequence/tss_and_centers_sorted(GM12878).txt'
    tss_and_centers_df.to_csv(sorted_tss_and_centers_path, sep='\t', index=False)


    # 输出结果到指定路径
    output_path = '/TF sequence/raw_data.txt'
    with open(output_path, 'w') as f:
        # 获取所有基因名称
        all_genes = set(overlaps_df['gene_name']).union(set(bed_df['gene']))

        # 分离有转录因子序列和没有转录因子序列的基因
        genes_with_tf = set(tss_and_centers_df['Gene'])
        genes_without_tf = all_genes - genes_with_tf

        # 输出有转录因子序列的基因
        for gene_name in genes_with_tf:
            # 获取该基因在overlaps_df中的信息
            gene_overlaps = overlaps_df[overlaps_df['gene_name'] == gene_name]
            tpm_value = gene_overlaps['TPM-GM12878'].iloc[0] if not gene_overlaps['TPM-GM12878'].empty else ''
            promoter_start_value = gene_overlaps['promoter_start'].iloc[0] if not gene_overlaps['promoter_start'].empty else ''
            promoter_end_value = gene_overlaps['promoter_end'].iloc[0] if not gene_overlaps['promoter_end'].empty else ''
            tf_chr_value = gene_overlaps['tf_chr'].iloc[0] if not gene_overlaps['tf_chr'].empty else ''
            strand_value = gene_overlaps['strand'].iloc[0] if not gene_overlaps['strand'].empty else ''

            # 获取该基因在tss_and_centers_df中的序列信息
            gene_tss_df = tss_and_centers_df[tss_and_centers_df['Gene'] == gene_name]
            sequence = ''.join(gene_tss_df['TF_Letter'].values)
            if gene_tss_df['Strand'].iloc[0] == '-':
                sequence = sequence[::-1]  # 如果基因位于负链，翻转序列

            # 写入文件
            f.write(f"{gene_name}\t{sequence}\t{tpm_value}\t{tf_chr_value}\t{promoter_start_value}\t{promoter_end_value}\t{strand_value}\n")

        # 输出没有转录因子序列的基因
        for gene_name in genes_without_tf:
            # 获取该基因在overlaps_df中的信息
            gene_overlaps = overlaps_df[overlaps_df['gene_name'] == gene_name]
            tpm_value = gene_overlaps['TPM-GM12878'].iloc[0] if not gene_overlaps['TPM-GM12878'].empty else ''
            promoter_start_value = gene_overlaps['promoter_start'].iloc[0] if not gene_overlaps['promoter_start'].empty else ''
            promoter_end_value = gene_overlaps['promoter_end'].iloc[0] if not gene_overlaps['promoter_end'].empty else ''
            tf_chr_value = gene_overlaps['tf_chr'].iloc[0] if not gene_overlaps['tf_chr'].empty else ''
            strand_value = gene_overlaps['strand'].iloc[0] if not gene_overlaps['strand'].empty else ''

            # 写入文件
            f.write(f"{gene_name}\t\t{tpm_value}\t{tf_chr_value}\t{promoter_start_value}\t{promoter_end_value}\t{strand_value}\n")

    print(f"Results have been written to {output_path}")

if __name__ == "__main__":
    main()