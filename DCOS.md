    1  export PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda-10.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    2  export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/extras/CUPTI/lib64:
    3  export CUDA_HOME=/usr/local/cuda-10.1
    4  export LIBRARY_PATH=/usr/local/cuda-10.1/lib64/stubs
    5  export NVIDIA_VISIBLE_DEVICES=all
    6  export NVIDIA_DRIVER_CAPABILITIES=compute,utility
    7  apt-get update -qq && apt-get install -y software-properties-common git nano
    8  apt-get install -y apt-utils cifs-utils  libboost-all-dev build-essential libssl-dev
    9  mkdir koko
   10  mount -t cifs -o user=,password= //10.51.2.245/tmp/yunshuang koko
   11  ls
   12  git clone git@github.com:YuanYunshuang/OpenCOOD.git
   13  ssh-keygen -t ed25519 -C "yuanyunshuang@163.com"
   14  cat /root/.ssh/id_ed25519.pub 
   15  git clone git@github.com:YuanYunshuang/OpenCOOD.git
   16  ls
   17  cd OpenCOOD/
   18  ls
   19  cd opencood/
   20  ls
   21  cd .. 
   22  rm -rf OpenCOOD/
   23  git clone git@github.com:YuanYunshuang/OpenCOOD.git -r
   24  git clone -r git@github.com:YuanYunshuang/OpenCOOD.git 
   25  git clone --recurse-submodules git@github.com:YuanYunshuang/OpenCOOD.git 
   26  cd OpenCOOD/opencood/
   27  ls
   28  git checkout fpv-rcnn
   29  git branch
   30  ls
   31  cd ..
   32  ls
   33  pip install -r requirements.txt 
   34  pip install numpy
   35  pip install matplotlib
   36  python
   37  cd spconv/
   38  python setup.py bdist_wheel
   39  ls
   40  cd ..
   41  git pull
   42  cd rm spconv/
   43  rm spconv/
   44  ls spconv
   45  rm -rf spconv/
   46  git clone https://github.com/traveller59/spconv.git --recursive
   47  cd spconv/
   48  python setup.py bdist_wheel
   49  pip install pccm
   50  python setup.py bdist_wheel
   51  cd ./dist/
   52  ls
   53  pip install spconv-2.1.21-py3-none-any.whl 
   54  cd ..
   55  ls
   56  cd pcdet/
   57  python setup.py develop
   58  cd ..
   59  ls
   60  ls koko
   61  ln -s /workspce/koko/experiments-output/opencood /workspace/OpenCOOD/opencood/logs
   62  git pull
   63  cd OpenCOOD/
   64  git pull
   65  python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/ciassd_early_fusion.yaml --model_dir ../koko/experiments-output/opencood/
   66  cd opencood/
   67  python ./tools/train.py --hypes_yaml ./hypes_yaml/ciassd_early_fusion.yaml --model_dir ../koko/experiments-output/opencood/
   68  cd ..
   69  python setup.py develop
   70  python opencood/utils/setup.py build_ext --inplace