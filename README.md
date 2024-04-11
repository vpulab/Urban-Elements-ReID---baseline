# Urban-Elements-ReID---baseline
In this repository, you can find instructions on how to download, configure, and run the baseline for the [Urban Elements ReID competition](https://www.kaggle.com/competitions/urban-reid-challenge/overview)

![](Urban-Elements-ReID---baseline
/logov3 (2).png)


## Download code and set up enviroment
To download the main code and set up the environment, please follow (at least) the first 3 steps of [Part Aware Transformer](https://github.com/liyuke65535/Part-Aware-Transformer).

## Modified codes
In order to use PAT for the [Urban Elements ReID competition](https://www.kaggle.com/competitions/urban-reid-challenge/overview) follow the next steps:

### 1) Download the UrbanElementsReID dataset
Download the UrbanElementsReID dataset from the section [Data](https://www.kaggle.com/competitions/urban-reid-challenge/data) in the Kaggle competition page and place it in the directory of your choice.

Once the dataset is downloaded run the `setup.sh` script over the dataset directoy in order to place the folders in the correct way. You can find this script in `/Codes/setup.sh`

```bash
cd "your data directory"
```

```bash
bash setup.sh
```

If needed give permissions to access the folders running
```bash
chmod +x "folder name"
```


### 2) Add the required files
Add to the folder `Part-Aware-Transformer/data/datasets/` the dataloaders and initialization files `UrbanElementsReID.py`, `UrbanElementsReID_test.py` and `__init__.py`.

Add to `Part-Aware-Transformer/config/` folder and set up the correspondig paths and configuration of `UrbanElementsReID_test.yml` and `UrbanElementsReID_train.yml` files.

Add to `Part-Aware-Transformer/utils/` the file `re_rankig.py`.

Add to `Part-Aware-Transformer/` the evaluation file `update.py`

### 3) Set up configuration files

Modify the configuration files `UrbanElementsReID_test.yml` and `UrbanElementsReID_train.yml` and set up your path to the data directory (DATASET:ROOT_DIR), pretrained model weigths (MODEL:PRETRAIN_PATH and TEST:WEIGHT) and output directory.

### 4) Train the model
In order to train the model first make sure that all the configuration settings and paths are correct. Then run the following line:

```bash
python train.py --config_file "config/UrbanElementsReID_train.yml"
```

### 5) Evaluation
To evaluate the results of the models use the script update.py to create the track_submission.csv and add the submission to Kaggle in order to obtain the obtained score. 

```bash
python update.py --config_file "config/UrbanElementsReID_test.yml" --track "path to store the files/track.txt"
```

### Acknowledgment 
Special thanks to liyuke65535 for the creation and publication of [Part Aware Transformer](https://github.com/liyuke65535/Part-Aware-Transformer) repository and congratulations for the excelent work.

This work has been supported by the Ministerio de Ciencia, Innovación y Universidades of the Spanish Government under project SEGA-CV (TED2021-131643A-I00)

