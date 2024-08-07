import pandas as pd
import numpy as np

#Read GSE89570 data
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

