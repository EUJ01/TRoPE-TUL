# TRoPE-TUL
The pytorch implementation version of TRoPETUL

The datasets are available at https://doi.org/10.34740/kaggle/dsv/15921592

The hyperparameters can be set from settings/local_test.json


To run the model: 
- clone the repo, then in terminal
- pip install -r requirements.txt
- make sure to put the datasets inside the corresponding dataset subfolder.
- data files required:
    - the POI.csv file.
    - the .npy embedding file (embedding creator file is inside the dataset folder.)
    - the .h5 file (creates an hd5 file used by the model. The file creator is inside the dataset folder.)
- make sure to login to your wandb account to see progress, type "wandb login" in your terminal and paste your API key when prompted.
- to run, type in terminal "python main.py".