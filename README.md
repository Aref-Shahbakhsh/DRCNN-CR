# DRCNN-CR
This code consists of 2 separate parts:
1: Preprocessing
2: Execute models
1: First, run the preprocessing files to get the dataset, clean it up and split it into a training set and a test set
First, you should run the Linux commands of the DNase phase:
The purpose of using DNase data is to extract pseudo cfDNA methylation data from DNA methylation, because DNA methylation is related to the tissue itself and cfDNA is released from the tissue into the blood, and DNase is a tissue component of DNA that is accessible and likely to be released later. into the blood.

 # Gathering Colorectal DNase footprint and hotspots data from GEO dataset which were aligned mostly using hg19 or GRch38 as their reference genome: 
GSM5214052, GSM5214053, GSM5214180, GSM5214420, GSM5215164, GSM5215620 (12 bed files)

# CrosspMap to convert genome coordinates between different assemblies such as hg18 (NCBI36) <=> hg19 (GRCh37):
```
!pip3 install CrossMap
!CrossMap bed hg38ToHg19.over.chain.gz '{insert DNase name file here}'.bed > '{new name of the converted file}'.bed
```

# BEDTools:
```
!wget https://github.com/arq5x/bedtools2/releases/download/v2.29.1/bedtools-2.29.1.tar.gz
!tar -zxvf bedtools-2.29.1.tar.gz
!cd bedtools2
!make
```
# Sorting each bed file and then merge within each bed file to reduce gaps:
```
!sort -k1,1 -k2n,2 '{the converted file name}'.bed > '{the sorted file name}'.bed
!bedtools merge -i '{the sorted file name}'.bed -d 10 > '{the merged file name}'.bed
```
# Concatenate all DNase bed files (two at the same time):
```
!cat '{insert one of merged files names here}'.bed '{insert one of the other merged files here}'.bed | sort -k 1,1 -k2,2n | bedtools merge > '{New name for the concatenated files}'.bed
```
# Now Run The TrainData.py File:
```
python TrainData.py
```
# Run bed tools:
```
!sort -k1,1 -k2,2n colon_dna_methylation.bed > colon_dna_methyl_sort.bed
!bedtools intersect -a colon_dna_methyl_sort.bed -b '{DNase concatenated files resulted from Dnase.py}'.bed -wa > intesect_colon.txt
```
# Now Extract the GSE89570_Processed_Data.txt.gz file Which is put on the Pre-processing Folder This file Downloaded From Below Link: 
```
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE89570
```
# Run First part of Test file with:
```
python TestData_Part1.py
```
# After you have performed the first part, start converting the interval-like position of cfDNA methylation into a point-like position of DNA methylation (CpG sites) by selecting the intervals that overlap with the CpG sites of DNA methylation:
```
!bedtools intersect -a '{cfDNA methylation positions}' -b '{DNA methylation positions}' -wa > cfDNA_intersect.bed
```
# Execute the last part of the test file to continue the conversion of interval positions to point-wist positions in addition to normalizing the cfDNA methylation data:
```
python TestData_Part2.py
```
# Now everything is ready to run Model, so first run the helper file to load the most important classes and libraries:
```
python helper.py
```
# Then Configure the model:
```
python model_configuration.py
```
# load Preproccesed Train data:
```
python TrainData.py
```
# For cross validation run:
```
python Cross-ValidationTraining.py
```
# And for Validation-reslut:
```
python Validation_results.py
```
# for the test phase, you should first load test data, then fine-tune the model and obtain the test result.
```
python TestData.py
```
```
python Fine-tuning.py
```
```
python testing_results.py
```
# Note: If you want to get a different result of this work instead of the best result, read the following instructions:
Normalization of cfDNA methylation data using the z-score method:
Replace line 40 with 41 in Test_Data_part2.py and run Test_Data_part2.py
Note: To obtain the z-score of the DNA methylation data, line 40 of this script should be applied to the final result of TrainData.py.
Note: To obtain non-normalized cfDNA methylation data, you can comment out and save lines 56 to 58 in the Test_Data_Part2 script.

Modes:
The previously explained method is for fine-tuning (cfDNA-methyl data not normalized or normalized with minmax or z-score) using pre-trained weights (methyl-DNA data normalized with z-score or minmax).
 Note: Data must be proportional to each other (except for non-normalized data). For example, if the DNA methylation was normalized with the z-score method to obtain the pre-trained weights, only the cfDNA methyl normalized with the z-score method can be used for fine-tuning. Tone and the final test used to determine the results.
Other modes:
- Without fine-tuning: all steps up to TestData.py are run, the fine-tuning.py code of the model is not executed and instead it is executed in testing_results.py, with the difference that the weights of the pre-trained best model are loaded.
Note: The training and test data should match, as in the previous example.

- Without using the pre-trained model:
All the helper.py and model_configuration.py codes should be executed, and then the TestData.py to testing_results.py codes, with the difference that in TestData.py, in the 14th line, instead of dividing the data by 50:50, the data is divided by 80:20. 80 for training and 20 for testing, as well as in Fine-tuning and testing_results, the first line is deleted and the initial learning rate in Fine-tuning.py is 0.000015.
Note: These steps can be run on only cfDNA methylation data in any mode (normalized or not).
