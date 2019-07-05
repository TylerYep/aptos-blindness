# APTOS 2019 Blindness Detection
#### Kaggle Competition

To get started, run:
```
kaggle competitions download -c aptos2019-blindness-detection
```

Rename ```train.csv``` to ```train-orig.csv``` and then run:
```
python split-csv.py
```
This gives you separate csvs for your train and dev sets.

Run ```python preprocess.py``` in order to get better images. You may need to uncomment the lines in that file.