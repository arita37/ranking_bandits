Data download link: 
https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0 

main_drive link: https://onedrive.live.com/?cid=8feadc23d838bda8&id=8FEADC23D838BDA8%21107&ithint=folder&authkey=!ACnoZZSZVfHPJd0

Download MQ2007.rar file and extract it 

It have 5 folds 

At first you have to preprocess the extracted data using bash script :
python data_preprocessing.py -train MQ2007/Fold2/train.txt -val MQ2007/Fold2/vali.txt -test MQ2007/Fold2/test.txt -out MQ2007/Fold2/chunked

In chunked folder we have all data in pickle 

I have shown example for Fold2, you can do for other analysis 

Then run main_file using below bash script :

python main_cpu.py --data_dir MQ2007/Fold1 --epochs_ext 10 --mid_dim 64 --nn_layers 2 --dropout 0.2 --batch_size 2 




