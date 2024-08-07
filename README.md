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
