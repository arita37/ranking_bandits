

# Setup Environment

   pyenv global 3.8.13 && python --version
   python -c 'import os; print(os)'
   pip3 list 

export CONDA_DIR="/workspace/miniconda"
export PATH=$CONDA_DIR/bin:$PATH
echo $PATH
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh -O miniconda.sh && \
    chmod a+x miniconda.sh && \
    bash ./miniconda.sh -b -p $CONDA_DIR && \
    rm ./miniconda.sh

conda init bash 
which python && which pip 


echo $PATH

conda create -n gstorm python==3.8.13
conda activate gstorm
which pip 


#### Pip install
      python -m  pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
        pip install dgl==1.0.4 -f https://data.dgl.ai/wheels-internal/repo.html

        ### Check
        pip install graphstorm --dry-run

        pip install graphstorm 


#### Sample test
        export WORKSPACE="$(pwd)/graphstorm"
        cd $WORKSPACE/examples

        python3 acm_data.py --output-path $WORKSPACE/ztmp/acm_raw --output-type raw_w_text





My environment has a GTX 1080 gpu. Before installing library below, you should have Nvidia driver and CUDA installed. 

- Python: 3.10.12
- OS: Ubuntu-22.04

[Notice] If you install CUDA with the latest version (12.x), it may have issues, since dgl may look for cudart.so.11

```
pip3 install graphstorm
pip3 install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install dgl==1.0.4+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
```

Then, configure SSH No-password login.

```
ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ''
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```
https://www.gitpod.io/docs/configure/workspaces/ports


Test it with ssh 127.0.0.1  

If it shows ```connection refused```, maybe the ssh server is not installed or started.

# Fethc Graphstorm Souce Code

```

git clone https://github.com/awslabs/graphstorm.git
```

[Notice] ```/mnt/h/graphsotrm``` is my workspace. I use ```$WORKSPACE``` to represent it in the following text, you should replace it with your own path.

# Prepare Raw Data

```
export WORKSPACE="$(pwd)/graphstorm"
ls $WORKSPACE
cd $WORKSPACE/examples

python3 acm_data.py --output-path $WORKSPACE/ztmp/acm_raw --output-type raw_w_text


```

The output will look like the screenshot below. It shows the information of author nodes, which indicates that the “text” column contains text feature.

<img src="./raw_data.png" width = "600" height = "400" alt="1" align=center />

# Construct Graph

```
python3 -m graphstorm.gconstruct.construct_graph \
           --conf-file $WORKSPACE/ztmp/acm_raw/config.json \
           --output-dir $WORKSPACE/ztmp/acm_nc \
           --num-parts 1 \
           --graph-name acm
```

<img src="./construct.png" width = "600" height = "100" alt="1" align=center />


Output will be like this:
<img src="./graph.png" width = "600" height = "400" alt="1" align=center />


# Launch GraphStorm Trainig without Fine-tuning BERT Models

```
        rm /tmp/ip_list.txt

touch /tmp/ip_list.txt
echo 127.0.0.1 > /tmp/ip_list.txt


 echo "22-arita37-rankingbandits-n9l5syb0gmg.ws-us105.gitpod.io" > /tmp/ip_list.txt




gp ports expose 22

gp ports visibility 22:public

https://22-arita37-rankingbandits-n9l5syb0gmg.ws-us105.gitpod.io


```

[Notice] If you only have one GPU, ```--num-trainers``` should be 1, or the trainning can't be launched.

```

python -c 'import graphstorm; print(graphstorm)'

python -m graphstorm.run.gs_node_classification \
        --workspace $WORKSPACE \
        --part-config $WORKSPACE/ztmp/acm_nc/acm.json \
        --ip-config /tmp/ip_list.txt \
        --num-trainers 1 \
        --num-servers 1 \
        --num-samplers 0 \
        --ssh-port 22 \
        --cf $WORKSPACE/graphstorm/examples/use_your_own_data/acm_lm_nc.yaml \
        --save-model-path $WORKSPACE/acm_nc/models \
        --node-feat-name paper:feat author:feat subject:feat


```

<img src="./train1_start.png" width = "800" height = "200" alt="1" align=center />

------------------------------------------------------------------

<img src="./train1_end.png" width = "600" height = "400" alt="1" align=center />

# Launch GraphStorm Trainig for both BERT and GNN Models

```
python3 -m graphstorm.run.gs_node_classification \
        --workspace $WORKSPACE \
        --part-config $WORKSPACE/ztmp/acm_nc/acm.json \
        --ip-config /tmp/ip_list.txt \
        --num-trainers 1 \
        --num-servers 1 \
        --num-samplers 0 \
        --ssh-port 22 \
        --cf $WORKSPACE/graphstorm/examples/use_your_own_data/acm_lm_nc.yaml \
        --save-model-path $WORKSPACE/acm_nc/both_models \
        --node-feat-name paper:feat author:feat subject:feat \
        --lm-train-nodes 10


```

<img src="./train2_start.png" width = "600" height = "300" alt="1" align=center />

------------------------------------------------------------------
<img src="./train2_end.png" width = "600" height = "400" alt="1" align=center />

# Only Use BERT Models

```
python3 -m graphstorm.run.gs_node_classification \
        --workspace $WORKSPACE/workspace \
        --part-config $WORKSPACE/ztmp/acm_nc/acm.json \
        --ip-config /tmp/ip_list.txt \
        --num-trainers 1 \
        --num-servers 1 \
        --num-samplers 0 \
        --ssh-port 22 \
        --cf $WORKSPACE/graphstorm/examples/use_your_own_data/acm_lm_nc.yaml \
        --save-model-path $WORKSPACE/acm_nc/only_bert_models \
        --node-feat-name paper:feat author:feat subject:feat \
        --lm-encoder-only \
        --lm-train-nodes 10


```

<img src="./train3_start.png" width = "600" height = "200" alt="1" align=center />

------------------------------------------------------------------
<img src="./train3_end.png" width = "600" height = "400" alt="1" align=center />
