```

Data download link: 
    https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0 

    main_drive link: https://onedrive.live.com/?cid=8feadc23d838bda8&id=8FEADC23D838BDA8%21107&ithint=folder&authkey=!ACnoZZSZVfHPJd0



Download MQ2007.rar file and extract it 
It have 5 folds 

###At first you have to preprocess the extracted data using bash script :

      python data_preprocessing.py -train MQ2007/Fold2/train.txt -val MQ2007/Fold2/vali.txt -test MQ2007/Fold2/test.txt -out MQ2007/Fold2/chunked

In chunked folder we have all data in pickle 
I have shown example for Fold2, you can do for other analysis 

Then run main_file using below bash script :

        python main_cpu.py --data_dir MQ2007/Fold1 --epochs_ext 10 --mid_dim 64 --nn_layers 2 --dropout 0.2 --batch_size 2 









#### Which command


1) how to Download the data ?
    https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0

    URL link


2)
   ### run this
   python main_cpu.py  --arg 1



3) 







# Data
The original dataset of MQ2007 can be downloaded from 


[here](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0). 


The preprocessed data files (after removing queries with no relevant documents) can be downloaded from [here](https://drive.google.com/file/d/13IPgtDq7YNiBoFGV_LXuxAPKIQLyAu_Y/view?usp=sharing). Place the downloaded files inside the directory *MQ2007*.

Search dataset 
   query ---> list of documents


Algo paper. RL algo for search.
https://arxiv.org/pdf/1910.10410.pdf 


python 3.8
pytorch 1.1 CPU

pip3 install torch==1.10.0+cpu --index-url  https://download.pytorch.org/whl/cpu/

https://download.pytorch.org/whl/cpu/torch-1.10.0%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256=4c4e1cd4cf4dceff2797f9fc167fe5da31682b4fa09fec1002d89114965bf680


#### 1) make the code runnable/testing the code
  Download the dataset and follow the steps.
  python main.py 
  python evaluate.py   --data_dir MQ2007/Fold   --model_file   best_models/

  epoch =1 
  smaller dataset size : 1000 rows to check.


#### Step 2
   use custom dataset  : I will provide.
   run training, run eval


# Train
The command for training the model on the mq2007 dataset using default parameters
```
python main.py 


```



# Evaluate 

For evaluating the model trained on the mq2007 dataset, use the below command
```
python evaluate.py   --data_dir MQ2007/Fold   --model_file   best_models/
```

it is working 

```