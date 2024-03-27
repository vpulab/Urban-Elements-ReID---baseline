# Urban-Elements-ReID---baseline

## Download code and set up enviroment
To download the main code and set up the enviroment please follow the instructions in [Part Aware Transformer](https://github.com/liyuke65535/Part-Aware-Transformer).

## Modified codes
In order to use PAT for the [Urban Elements ReID competition](https://www.kaggle.com/competitions/urbam-reid-challenge/overview) follow the next steps:

### 1) Download the UrbanElementsReID dataset
Download the UrbanElementsReID dataset from the section [Data](https://www.kaggle.com/competitions/urbam-reid-challenge/data) in the Kaggle competition page and place it in the directory of your choice.

### 2) Add the necessary files
Add to the folder `Part-Aware-Transformer-main/data/datasets/` the dataloaders and initialization files `UrbanElementsReID.py`, `UrbanElementsReID_test.py` and `__init__.py`.

Add to `Part-Aware-Transformer-main/config/` folder and set up the correspondig paths and configuration of `UrbanElementsReID_test.yml` and `UrbanElementsReID_train.yml` files.

Add to `Part-Aware-Transformer-main/utils/` the file `re_rankig.py`.

Add to `Part-Aware-Transformer-main/` the evaluation file `update.py`

### 3) Set up configuration files

Modify the configuration files to set up the path to the data directory.

### 4) Train the model
In order to train the model first make sure that all the condifuration settings and paths are correct. Then run the following line:

```bash
python train.py --config_file "config/UrbanElementsReID_train.yml"
```

### 5) Evaluation
To evaluate the results of the models use the script update.py to create the track_submission.csv and add the submission to Kaggle in order to obtain the obtained score. 

```bash
python update.py --config_file "config/UrbanElementsReID_test.yml" --track "path to store the track.txt and track_submission.csv"
```

### Acknowledgment 
Special thanks to liyuke65535 for the creation and publication of [Part Aware Transformer](https://github.com/liyuke65535/Part-Aware-Transformer) repository and congratulations for the excelent work.
