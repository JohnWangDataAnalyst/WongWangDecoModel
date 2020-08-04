# WongWangDecoModel
# two states of WongWangDecol model with BOLD dynamic 
## senistive model is used for gradient calculation
## statistics R is used to identify the uniqueness of density of model parameters by fitting data multiple times

scripts fold: qsub : shell script 
qsub -V -cmd qsub_subID.sh
