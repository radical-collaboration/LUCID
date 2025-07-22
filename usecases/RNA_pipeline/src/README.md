# LUCID_Novel-lncRNA

This folder only holds the radical pilot based execution script. the following is just an example execution fo the given pipeline

## 1. Novel LncRNA Detection Pipeline
### 1.1. Top100.py
*Example command prompt
python Top100.py -i ./Data/LoRA_BERT/d0W1R1.csv -f ./Data/LoRA_BERT/d0W1R1.fasta -o d0W1R1_top100.fasta

### 1.2. BLAST_Target.py
*Example command prompt
python BLAST_Target.py -i ./Data/d0W1R1_BLAST.txt -o d0W1R1_novel.csv


## 2. Reverse: Calculating TPM/Counts from Target Sequence

### 2.1. TargetSequence_location.py
*Example command prompt
python TargetSequence_location.py -i ./Data/novel_lncRNA -o gene_location.csv


### 2.2. Location_StringTie.py
*Example command prompt
python Location_StringTie.py -i gene_location.csv -s ./Data/Strintie_output
