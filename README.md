# DRCNN-CR
This Code Contains 2 Separatly parts:
1: Preprocessing
2: Run Models
1: To begin with First Run The Preprocessing Files To Get Data-Set, Clear and Prepare into Train and Test set
First You Should Run DNase Phase Linux Commands:
The purpose of using DNase data is to extract pseudo-cfDNA methylation data from DNA methylation, because DNA methylation is related to the tissue itself, and cfDNA is shed from the tissue into the blood, and DNase is a tissue part of DNA that is accessible and probably later. be released in blood.

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
# After runnig the first part, begin to turn interval-type position of cfDNA methylation into point-type position of DNA methylation (CpG sites) by choosing the intervals which overlaps the CpG sites on DNA methylation:
```
!bedtools intersect -a '{cfDNA methylation positions}' -b '{DNA methylation positions}' -wa > cfDNA_intersect.bed
```
# Run Last Part of Test file to continue converting interval positions to point-wist positions alongside normalizing the cfDNA methylation data‌:
```
python TestData_Part2.py
```
# Now every thing is Ready for runing Model so first run the helper file to load essential class and library!:
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
# for Test phase you should first load Test Data then Fine-tune the model and get the test result.
```
python TestData.py
```
```
python Fine-tuning.py
```
```
python testing_results.py
```
# Note: if you want to get other result of this paper instead of best result read the foallowing guids:
Normalization of cfDNA methylation data with z-score method:
Replace line 40 with 41 in Test_Data_part2.py and run Test_Data_part2.py
Note: To get the z-score of DNA methylation data, line 40 of this script should be used on the final result of TrainData.py.
Note: To obtain unnormalized cfDNA methylation data, lines 56 to 58 in the Test_Data_Part2 script can be commented out and saved.

Modes:
The previous one that we explained is for fine-tuning (cfDNA methyl data not normalized or normalized with minmax or z-score) using pre-trained weights (methyl DNA data normalized with z-score or minmax).
 Note: The data must be proportional to each other (except for non-normalized data). For example, if DNA methylation normalized by the z-score method is used to obtain the pretrained weights, only cfDNA methyl normalized by the z-score method can be used for fine-tuning. Tone and the final test used to get the results.

Other modes:
- Without fine-tuning: All the steps until TestData.py are passed, the fine-tuning.py code of the model is not executed and instead it is executed in testing_results.py, with the difference that the weights of the pre-trained best model are loaded.
Note: The training and test data should match, as in the previous example.

- Without using the pre-trained model:
All the helper.py and model_configuration.py codes should be executed, and then the TestData.py to testing_results.py codes, with the difference that in TestData.py, in the 14th line, instead of dividing the data by 50:50, the data is divided by 80:20. 80 for training and 20 for testing, as well as in Fine-tuning and testing_results, the first line is deleted and the initial learning rate in Fine-tuning.py is 0.000015.
Note: These steps can be run on only cfDNA methylation data in any mode (normalized or not).
