# Guideline(Installation)  

## Dataset  

We follow the same pre-processing strategy described in the below link.

[https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md](https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md)


You should put datasets into relevant directory.  
* Visual Genome => Datasets/VG  
```bash  
.   
├── Datasets    
│       └── VG    
│           ├── image_data.json   
│           ├── VG-SGG-with-attri.h5
│           ├── VG-SGG-dicts-with-attri.json
│           ├── Category_Type_Info.json
│           └── VG_100k
│                   └── *.png
```  
* Open Images V6 => Datasets/OI-V6 
```bash  
.   
├── Datasets    
│       └── OI-v6    
│           ├── Category_Type_Info.json   
│           ├── annotations   
│           │       ├── categories_dict.json
│           │       ├── vrd-test-anno.json
│           │       ├── vrd-val-anno.json
│           │       └── vrd-train-anno.json
│           └── images     
│                  └── *.png
```  

* Open Images V4 => Datasets/OI-V4  
```bash  
.   
├── Datasets    
│       └── OI-v4 
│           ├── Category_Type_Info.json   
│           ├── annotations   
│           │       ├── categories_dict.json
│           │       ├── vrd-val-anno.json
│           │       └── vrd-train-anno.json
│           └── images     
│                  └── *.png
```  

## Pretrained Faster R-CNN  

We employ the same pretrained Faster R-CNN module corresponding to [BGNN](https://github.com/SHTUPLUS/PySGG)  

* Faster R-CNN
    * [vg](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EQIy64T-EK9Er9y8kVCDaukB79gJwfSsEIbey9g0Xag6lg?e=wkKHJs)  
    * [oi-v6](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfGXxc9byEtEnYFwd0xdlYEBcUuFXBjYxNUXVGkgc-jkfQ?e=lSlqnz)  
    * [oi-v4](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EcxwkWxBqUdLuoP58vnUyMABR2-DC33NGj13Hcnw96kuXw?e=NveDcl)  

You should download the  faster r-cnn model and put the downloaded path in **shell/\*.sh**

## Package Install

``` python  

conda create -n hetsgg python=3.7.7

conda activate hetsgg

conda install -y ipython scipy h5py

pip install ninja yacs cython matplotlib tqdm opencv-python overrides gpustat gitpython ipdb graphviz tensorboardx termcolor scikit-learn==0.23.1

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch

pip install torch-scatter==2.0.7 torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.0+cu110.html

pip install torch-sparse -f https://data.pyg.org/whl/torch-1.7.0+cu110.html

pip install torch-geometric

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

cd ..

git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

cd ..

python setup.py build develop

```  

## Implementation  

You should train the **HetSGG** model in **shell/** directory.


* Train  
``` python  
## SGCls
bash shell/hetsgg_train_sgcls_vg.sh  

## SGGen  
bash shell/hetsgg_train_sggen_vg.sh
```  

* Test  
``` python  
# You should put the model checkpoint name on .sh
bash shell/hetsgg_test.sh
```  
### HetSGG++

You should change the value of **model_config** in shell/\*.sh files to *relHetSGGp_vg*.

```python  
export model_config="relHetSGGp_vg"
``` 
