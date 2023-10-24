
# Setup Environment
        # pyenv global 3.8.13 && python --version
        # python -c 'import os; print(os)'
        # pip3 list 
        pyenv global system ### remove  


#### Setup Conda 
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

        ls  
        /workspace/miniconda/envs/gstorm/bin/pip list

        echo $PYTHONPATH



#### Pip install
      python -m  pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
        pip install dgl==1.0.4 -f https://data.dgl.ai/wheels-internal/repo.html

        ### Check
        pip install graphstorm --dry-run

        pip install graphstorm 
        pip list 


#### Sample test
        export WORKSPACE="$(pwd)/graphstorm"
        cd $WORKSPACE/examples

        python3 acm_data.py --output-path $WORKSPACE/ztmp/acm_raw --output-type raw_w_text



### SSH server install
        sudo apt-get install openssh-server
        sudo /etc/init.d/ssh restart




##### My environment has a GTX 1080 gpu. 
Before installing library below, you should have Nvidia driver and CUDA installed. 

    - Python: 3.10.12
    - OS: Ubuntu-22.04


####  CUDA with the latest version (12.x), 
it may have issues, since dgl may look for cudart.so.11

    pip3 install graphstorm
    pip3 install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip3 install dgl==1.0.4+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html


### SSH server install
        sudo apt-get install openssh-server
        sudo /etc/init.d/ssh restart




#### Then, configure SSH No-password login.
        ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ''
        cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

        https://www.gitpod.io/docs/configure/workspaces/ports


        Test it with ssh 127.0.0.1  
        If it shows ```connection refused```, maybe the ssh server is not installed or started.



#### Fethc Graphstorm Souce Code
        git clone https://github.com/awslabs/graphstorm.git

        [Notice] ```/mnt/h/graphsotrm``` is my workspace. I use ```$WORKSPACE``` to represent it in the following text, you should replace it with your own path.


# Prepare Raw Data
        export WORKSPACE="$(pwd)/graphstorm"
        ls $WORKSPACE
        cd $WORKSPACE/examples

        python3 acm_data.py --output-path $WORKSPACE/ztmp/acm_raw --output-type raw_w_text



The output will look like the screenshot below. It shows the information of author nodes, which indicates that the “text” column contains text feature.

<img src="./raw_data.png" width = "600" height = "400" alt="1" align=center />

##### Construct Graph
        python3 -m graphstorm.gconstruct.construct_graph \
                --conf-file $WORKSPACE/ztmp/acm_raw/config.json \
                --output-dir $WORKSPACE/ztmp/acm_nc \
                --num-parts 1 \
                --graph-name acm



# Launch GraphStorm Trainig without Fine-tuning BERT Models
        rm /tmp/ip_list.txt

### SSH server install
        sudo apt-get install -y openssh-server
        sudo /etc/init.d/ssh restart

        ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ''
        cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

        touch /tmp/ip_list.txt
        echo 127.0.0.1 >  /tmp/ip_list.txt

### kill port
sudo lsof -i :22
sudo kill -9 4535



[Notice] If you only have one GPU, ```--num-trainers``` should be 1, or the trainning can't be launched.

        source activate gstorm

        python -c 'import graphstorm; print(graphstorm)'

        python -m graphstorm.run.gs_node_classification \
                --workspace $WORKSPACE \
                --part-config $WORKSPACE/ztmp/acm_nc/acm.json \
                --ip-config  "/tmp/ip_list.txt" \
                --num-trainers 1 \
                --num-servers 1 \
                --num-samplers 0 \
                --ssh-port 22 \
                --cf $WORKSPACE/examples/use_your_own_data/acm_lm_nc.yaml \
                --save-model-path $WORKSPACE/acm_nc/models \
                --node-feat-name paper:feat author:feat subject:feat




##### Launch GraphStorm Trainig for both BERT and GNN Models
        python3 -m graphstorm.run.gs_node_classification \
                --workspace $WORKSPACE \
                --part-config $WORKSPACE/ztmp/acm_nc/acm.json \
                --ip-config /tmp/ip_list.txt \
                --num-trainers 1 \
                --num-servers 1 \
                --num-samplers 0 \
                --ssh-port 22 \
                --cf $WORKSPACE/examples/use_your_own_data/acm_lm_nc.yaml \
                --save-model-path $WORKSPACE/acm_nc/both_models \
                --node-feat-name paper:feat author:feat subject:feat \
                --lm-train-nodes 10


##### Only Use BERT Models
        python3 -m graphstorm.run.gs_node_classification \
                --workspace $WORKSPACE/workspace \
                --part-config $WORKSPACE/ztmp/acm_nc/acm.json \
                --ip-config /tmp/ip_list.txt \
                --num-trainers 1 \
                --num-servers 1 \
                --num-samplers 0 \
                --ssh-port 22 \
                --cf $WORKSPACE/examples/use_your_own_data/acm_lm_nc.yaml \
                --save-model-path $WORKSPACE/acm_nc/only_bert_models \
                --node-feat-name paper:feat author:feat subject:feat \
                --lm-encoder-only \
                --lm-train-nodes 10




INFO:root:_model_encoder_type: rgcn
INFO:root:_backend: gloo
INFO:root:_verbose: False
INFO:root:_fanout: 50,50
INFO:root:_num_layers: 2
INFO:root:_hidden_size: 256
INFO:root:_use_mini_batch_infer: False
INFO:root:_restore_model_path: None
INFO:root:_save_model_path: /workspace/ranking_bandits/graphstorm/acm_nc/models
INFO:root:_save_embeds_path: /tmp/acm_nc/embeds
INFO:root:_dropout: 0.0
INFO:root:_lr: 0.0001
INFO:root:_lm_tune_lr: 0.0001
INFO:root:_num_epochs: 20
INFO:root:_batch_size: 1024
INFO:root:_wd_l2norm: 0
INFO:root:_alpha_l2norm: 0.0
INFO:root:_num_bases: -1
INFO:root:_use_self_loop: True
INFO:root:_sparse_optimizer_lr: 1e-2
INFO:root:_use_node_embeddings: False
INFO:root:_target_ntype: paper
INFO:root:_label_field: label
INFO:root:_multilabel: False
INFO:root:_num_classes: 14
INFO:root:_task_type: node_classification
INFO:root:_logging_level: info
INFO:root:_profile_path: None
INFO:root:_construct_feat_ntype: None
INFO:root:_node_feat_name: ['paper:feat', 'author:feat', 'subject:feat']
INFO:root:_ip_config: /tmp/ip_list.txt
INFO:root:_part_config: /workspace/ranking_bandits/graphstorm/ztmp/acm_nc/acm.json



#### Config

{
    "nodes": [
        {
            "node_type": "author",
            "format": {
                "name": "parquet"
            },
            "files": [
                "/workspace/ranking_bandits/graphstorm/ztmp/acm_raw/nodes/author.parquet"
            ],
            "node_id_col": "node_id",
            "features": [
                {
                    "feature_col": "feat",  "feature_name": "feat"
                },
                {
                    "feature_col": "text",  "feature_name": "text",
                    "transform": {
                        "name": "tokenize_hf",
                        "bert_model": "bert-base-uncased",
                        "max_seq_length": 16
                    }
                }
            ]
        },
        {
            "node_type": "paper",
            "format": {
                "name": "parquet"
            },
            "files": [
                "/workspace/ranking_bandits/graphstorm/ztmp/acm_raw/nodes/paper.parquet"
            ],
            "node_id_col": "node_id",
            "features": [
                {
                    "feature_col": "feat",   "feature_name": "feat"
                },
                {
                    "feature_col": "text",   "feature_name": "text",
                    "transform": {
                        "name": "tokenize_hf",
                        "bert_model": "bert-base-uncased",
                        "max_seq_length": 16
                    }
                }
            ],
            "labels": [
                {
                    "label_col": "label",
                    "task_type": "classification",
                    "split_pct": [
                        0.8,
                        0.1,
                        0.1
                    ]
                }
            ]
        },
        {
            "node_type": "subject",
            "format": {
                "name": "parquet"
            },
            "files": [
                "/workspace/ranking_bandits/graphstorm/ztmp/acm_raw/nodes/subject.parquet"
            ],
            "node_id_col": "node_id",
            "features": [
                {
                    "feature_col": "feat",  "feature_name": "feat"
                },
                {
                    "feature_col": "text", "feature_name": "text",
                    "transform": {
                        "name": "tokenize_hf",
                        "bert_model": "bert-base-uncased",
                        "max_seq_length": 16
                    }
                }
            ]
        }
    ],
    "edges": [
        {
            "relation": [  "author", "writing", "paper" ],
            "format": { "name": "parquet" }, 
             "files": [
                "/workspace/ranking_bandits/graphstorm/ztmp/acm_raw/edges/author_writing_paper.parquet"
            ],
            "source_id_col": "source_id",
            "dest_id_col": "dest_id"
        },

        {
            "relation": [ "paper", "cited", "paper" ], 
            "format": { "name": "parquet" }, 
            "files": [ "/workspace/ranking_bandits/graphstorm/ztmp/acm_raw/edges/paper_cited_paper.parquet"
            ],
            "source_id_col": "source_id",
            "dest_id_col": "dest_id"
        },
        {
            "relation": [ "paper", "citing", "paper" ],
            "format": { "name": "parquet" },
            "files": [ "/workspace/ranking_bandits/graphstorm/ztmp/acm_raw/edges/paper_citing_paper.parquet" ],
            "source_id_col": "source_id",
            "dest_id_col": "dest_id"
        },
        {
            "relation": [ "paper", "is-about", "subject" ], 
            "format": {  "name": "parquet"
            },
            "files": [
                "/workspace/ranking_bandits/graphstorm/ztmp/acm_raw/edges/paper_is-about_subject.parquet"
            ],
            "source_id_col": "source_id",
            "dest_id_col": "dest_id"
        },
        {
            "relation": [ "paper", "written-by", "author" ], "format": {
                "name": "parquet"
            },
            "files": [
                "/workspace/ranking_bandits/graphstorm/ztmp/acm_raw/edges/paper_written-by_author.parquet"
            ],
            "source_id_col": "source_id",
            "dest_id_col": "dest_id"
        },
        {
            "relation": [
                "subject",
                "has",
                "paper"
            ],
            "format": {
                "name": "parquet"
            },
            "files": [
                "/workspace/ranking_bandits/graphstorm/ztmp/acm_raw/edges/subject_has_paper.parquet"
            ],
            "source_id_col": "source_id",
            "dest_id_col": "dest_id"
        }
    ]
}