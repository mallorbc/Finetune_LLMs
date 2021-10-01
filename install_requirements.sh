#!/bin/bash
#general requirements
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y
pip install datasets
pip install transformers
#need to drastically lower requirements
git clone https://github.com/microsoft/DeepSpeed -b v0.5.3
cd DeepSpeed
DS_BUILD_OPS=1 pip install .
ds_report
