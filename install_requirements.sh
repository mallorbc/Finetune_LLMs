#!/bin/bash
#general requirements
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y
pip install datasets==1.16.1
pip install transformers==4.14.1
#need to drastically lower requirements
git clone https://github.com/microsoft/DeepSpeed -b v0.5.8
cd DeepSpeed
DS_BUILD_OPS=1 pip install .
ds_report
