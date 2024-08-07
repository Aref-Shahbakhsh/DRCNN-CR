# DRCNN-CR
This Code Contains 2 Separatly parts:
1: Preprocessing
2: Run Models
You Should First Run The Preprocessing Files To Get Data-Set, Clear and Prepare into Train and Test set
First You Should Run This Linux Command to Do The Preprocessing Phase:
 # Gathering Colorectal DNase foorprint and hotspots data from GEO dataset which were aligned mostly using hg19 or GRch38 as their reference genome: 
GSM5214052, GSM5214053, GSM5214180, GSM5214420, GSM5215164, GSM5215620 (12 bed files)

# CrosspMap to convert genome coordinates between different assemblies such as hg18 (NCBI36) <=> hg19 (GRCh37)
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
# Concatenate all DNase bed files (two at the same time)
```
!cat '{insert one of merged files names here}'.bed '{insert one of the other merged files here}'.bed | sort -k 1,1 -k2,2n | bedtools merge > '{New name for the concatenated files}'.bed
```
