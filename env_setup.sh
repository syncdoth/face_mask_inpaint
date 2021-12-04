env_name="ref_mask"
py_ver="3.8"
cuda_ver="10.1"

conda create -n $env_name python=$py_ver
conda activate $env_name

################### UGCPU specific Settings #######################
# ugcpu4 and 5 have gcc versions 4.x, which is incompatible with the c++ source
# used for stylegan2 in this repo. We need gcc version >=5.
# uncomment the following lines if you are running on ugcpu4 or 5.
################### Uncomment Below ###############################

# conda install -y gcc-5 -c psi4

# CONDA_DIR="~/miniconda3"   ## SET THIS TO YOUR CORRECT DIRECTORY!
# cd $CONDA_DIR/envs/$env_name/bin/
# cp x86_64-unknown-linux-gnu-c++ x86_64-conda-linux-gnu-c++
# cp x86_64-unknown-linux-gnu-gcc x86_64-conda-linux-gnu-cc

##################################################################

# basic
conda install -y pandas scikit-learn scipy matplotlib jupyter tqdm tensorboard

# deep learning
conda install -y pytorch torchvision cudatoolkit=$cuda_ver -c pytorch

# conda-forge
conda install -y wandb -c conda-forge

# pip
pip install opencv-python pytorch-msssim

# c++ related
conda install -y ninja
