import pandas as pd
import numpy as np

#You should Download GSE89570 data from GEO website
df = pd.read_csv("GSE89570_Processed_Data.txt", sep = '\t')

#Seperating Chromosome, Start, and End of each position
df[['Chrom', 'Start', 'End']] = df.pos.str.split(pat='_',expand=True)
df['Chrom'] = 'chr' + df['Chrom'].astype(str)
df = df.drop(columns = 'pos', axis = 1)
temp_cols=df.columns.tolist()
new_cols=temp_cols[-3:] + temp_cols[:-3]
df=df[new_cols]
df["Start"] = df["Start"].astype(int)
df["End"] = df["End"].astype(int)

#Turning interval-type position of cfDNA methylation into point-type position of DNA methylation (CpG sites)
!bedtools intersect -a '{cfDNA methylation positions}' -b '{cfDNA methylation positions}' -wa > cfDNA_intersect.bed

cfDNA_single = pd.read_csv("/home/fmahdavi/Colon/cfDNA/cfDNA_intersect.bed",sep = "\t",header = None)
cfDNA_single = np.array(cfDNA_single)

list_position_final = []
for i in cfDNA_single:
    temp = df[(df["Chrom"] == i[0]) & (df["Start"] <= i[1]) & (df["End"] >= i[2])].reset_index(drop = True)
    if len(temp) >= 1:
        list_position_final.append(i[0:3].tolist() + list(temp.loc[0])[3:])
        
position_cfDNA = pd.DataFrame(list_position_final)
position_cfDNA.columns = df.columns

#Extracting normal and cancerous colorectal cfDNA methylaion samples
colon_cancer_idx = list(range(192, 329))+list(range(396, 402))+list(range(597, 602))
colon_benign_idx = list(range(329, 396))
colon_idx = colon_cancer_idx + colon_benign_idx
list_colon = [l[i] for i in colon_idx]
df_cfDNAmethyl = position_cfDNA.loc[:,list(position_cfDNA.columns)[0:4] + list_colon]

#Setting the final testing data following by two different normalization method
df_DNAmethyl = pd.read_csv("intesect_colon.txt",sep = "\t")
cols = df_DNAmethyl.columns
df_DNAmethyl_cols = df_DNAmethyl[cols[0:3]]

df_norm_cfdna_tot = pd.merge(df_DNAmethyl_cols, df1, on=['Chrom', 'Start', 'End'], how='left')
df_norm_cfdna_tot = df_norm_cfdna_tot.drop_duplicates(subset=['Chrom', 'Start', 'End'], keep='first')
df_norm_cfdna_tot = df_norm_cfdna_tot.replace(np.nan, 0)
df_norm_cfdna_tot = df_norm_cfdna_tot.reset_index(drop=True)

df_norm_cfdna_tot["ID"] = df_norm_cfdna_tot["Chrom"] + "-" + df_norm_cfdna_tot["Start"].astype(str) + "-" 
+ df_norm_cfdna_tot["End"].astype(str)
list_id = list(df_norm_cfdna_tot["ID"])
list_index = (df_norm_cfdna_tot.columns)[4:-1]
test_data = df_norm_cfdna_tot[list_index].T

df_norm = test_data.copy()
for col in test_data.columns:
    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())  #Min-Max normalization
#    df_norm[col] = (df_norm[col] - df_norm[col].mean())/df_norm[col].std(ddof=0)   #Z-score normalization

df_norm = df_norm.replace(np.nan, 0)
df_norm.to_csv("cfdna_colon_test_minmax.txt",sep = "\t", header = True, index = False)
