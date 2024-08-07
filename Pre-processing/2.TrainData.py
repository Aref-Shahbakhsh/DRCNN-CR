import pandas as pd
import numpy as np

df = pd.read_csv('TCGA.COAD.sampleMap%2FHumanMethylation450', sep='\s+') #Normal samples (GSE132804, GSE193535) are added to TCGA samples 
df.rename(columns = {'sample':'#id'}, inplace = True)

df_id = pd.read_csv('probeMap%2FilluminaMethyl450_hg19_GPL16304_TCGAlegacy', sep='\t')

result = pd.merge(df_id, df, on="#id")
result = result.drop(columns = ['strand'], axis = 1)
result = result.rename(columns={"chrom": "Chrom", "chromStart": "Start", "chromEnd": "End"})
cols = result.columns.tolist()
cols = cols[2:5] + cols[0:2] + cols[5:]
result = result[cols]

result = result[result['Chrom'].notna()]
result = result.replace(np.nan, 0)

header = list(result.columns)

result.to_csv('colon_dna_methylation.bed', index=False, header=False, sep = '\t')

!sort -k1,1 -k2,2n colon_dna_methylation.bed > colon_dna_methyl_sort.bed
!bedtools intersect -a colon_dna_methyl_sort.bed -b '{DNase concatenated files resulted from Dnase.py}'.bed -wa > intesect_colon.txt
