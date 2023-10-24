#export PYTHONPATH="$(pwd)"
# echo "PYTHONPATH $PYTHONPATH"

###Shorten
export PS1='\W$ '

#### Custom bashtc for gitpod
## vim /home/gitpod/.bashrc


### source init.sh

echo "init"
function init() {

    echo " export PS1='\W$ ' " >> ~/.bashrc
    echo ' export CONDA_DIR="/workspace/miniconda" '  >> ~/.bashrc
    echo ' export PATH=$CONDA_DIR/bin:$PATH '         >> ~/.bashrc


    export PS1='\W$ '
    export CONDA_DIR="/workspace/miniconda"
    export PATH=$CONDA_DIR/bin:$PATH 

    # conda init bash

    which conda && which pip && which python
    python -c 'import json; print(json)'

    tail -n 5 ~/.bashrc

    echo "   vim /home/gitpod/.bashrc "

    conda env list

}


echo "conda_install"
function conda_install() {

        export CONDA_DIR="/workspace/miniconda"
        export PATH=$CONDA_DIR/bin:$PATH
        echo $PATH
        wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh -O miniconda.sh && \
        chmod a+x miniconda.sh && \
        bash ./miniconda.sh -b -p -y $CONDA_DIR && \
        rm ./miniconda.sh

        echo "### remove pyenv Default"
        pyenv global system  

        # conda init bash 
        which python && which pip 

}



function killport() {

   sudo lsof -i :22
   echo "    sudo lsof -i :22" 
   echo "sudo kill -9 4535"



}

##### Python install
#   pyenv install 3.8.13 && pyenv global 3.8.13 && python --version
#   python -c 'import os; print(os)'
#   pip install --upgrade dryrun




# List Python versions in the terminal:

# pyenv install --list | grep " 3\.[678]"
# Install Python version if not in list:

# pyenv install 3.8.6
# Create a virtual env with a Python version:

# pyenv virtualenv 3.8.6 project1
# List versions of virtual environments:

# pyenv versions
# Activate a virtual version:

# pyenv activate project1


# git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

