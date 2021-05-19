# Evaluating Deep Learning Models for Subseasonal Forecasting over India

![tmp2m visualization](visualisations/tmp2m.png?raw=true "tmp2m visualisation")

Code for reproducing the results in my 2021 dissertation @TheUniversityOfSheffield

Please execute all scripts from the base directory of the repository, i.e., the directory in which README.md is located

## Environment and packages
The code was implemented using python 3.7.10 using Anaconda 4.7.10 and uses the following python 3.7 packages
  - **netCDF4**: 1.5.6
  - **numpy**: 1.19.2
  - **pandas**: 1.2.3
  - **xarray**: 0.17.0
  - **cdo**: 1.5.4
  - **scipy**: 1.6.2
  - **scikit-learn**: 0.24.1
  - **hdf5**: 1.10.6
  - **pytables**: 3.6.1
  - **tensorflow-gpu**: 2.4.1
  - **matplotlib**: 3.4.1
  - **wandb**: 0.10.30

  ## Installation of packages
  The packages can be installed directly into a conda environment by running the following commands  
  ```conda install -c conda-forge netCDF4```  
  
  ```conda install numpy```  
  
  ```conda install pandas```  
  
  ```conda install xarray```  
  
  ```conda install -c conda-forge cdo```  
  
  ```conda install scipy```  
  
  ```conda install scikit-learn```  
  
  ```conda install -c conda-forge netCDF4```  
  
  ```conda install -c conda-forge hdf5=1.10.6```  
  
  ```conda install -c conda-forge pytables```  
  
  ```pip install tensorflow-gpu```  
  
  ```pip install matplotlib```  
  
  ```pip install wandb```  
  
  or
  
  ```conda install --file requirements.txt```
  
  ## Downloading files
  
  ### SubseasonalRodeo
  Run ```./download/SubseasonalRodeo.sh```
  
  ### IMD
  Run ```./downlaod/imd-daily.sh```
  
  ### IMDAA
  1) Navigate to https://rds.ncmrwf.gov.in/dashboard/download
  2) Submit a request for the following variables:
    - IMDAA Daily Single Level Dataset (Years: ALL, Months: ALL, Days: ALL, Formats: NetCDF, Whole Area)
      - Surface Pressure (00Z-00Z) 
      - Mean Sea Level Pressure (00Z-00Z)
      - 2m Relative Humidity (00Z-00Z)
    - IMDAA IMDAA 1-Hourly Single Level Dataset
      - Surface Pressure (Years: 1979 and 2019, Months: ALL, Days: ALL, Formats: NetCDF, Whole Area)
      - Mean Sea Level Pressure (Years: 1979, Months: ALL, Days: ALL, Formats: NetCDF, Whole Area)
      - 2m Relative Humidity (Years: 1979 and 1983, Months: ALL, Days: ALL, Formats: NetCDF, Whole Area)
  3) Download the script which is emailed to you after the dataset is prepared and place it in data/ncmrwf/
  4) Run download/ncmrwf-daily.sh
  
  ## Interpolating files
  ```cd data```  
  
  ```./imd-interpolate.sh```  
  
  ```./ncmrwf-interpolate.sh```
  
  ## Calculate climatologies
  ```cd data```  
  
  ```./climatology.sh```
  
  ## Preprocess
  ```python preprocess.py -i``` (for Indian dataset)  
  
  ```python preprocess.py -r``` (for Rodeo dataset)
  
  ## Train
  Configure hyperparameter dict in train.py  
  After setting the hyperparameters, run
  
  ```python train.py```
  
  ## Latex tables
  
  - Configure 10 sweeps per sweep.yml file in sweeps folder
  - Create graph of architecture vs test_cosine_similarity on wandb
  - Export graph as csv
  - Place in results/
  - ```python results.py``` to produce the latex tables. Resulting latex tables will be saved in results/tables/
 
  ## Cache files
  ```
  # train.py
  31 cache_dir = os.path.join('/', 'fastdata', 'aca18mms')
  32 os.environ['WANDB_DIR'] = cache_dir
  33 os.environ['WANDB_IGNORE_GLOBS'] = '*.h5'
  ```
  By default the training files are saved in the temporary directory on bessemer.
  You can change the storage location by changing line 31 in train.py
  Wandb also ignores and does not upload the model to the server to save space. Comment line 33 to modify that.

  ## Author
  Manas Sarpatwar
  19/05/2021
  
  ## Acknowledgements
  I would like to thank my supervisor, Dr. Aditya Gilra for his kind help and support throughout this project. I would also like to thank Dr. Sreejith O.P and Dr. Rajib     Chattopadhyay from the India Meteorological Department, Pune, India for providing me their expertise on climate and subseasonal forecasting. Lastly, I would like to thank my parents and my sister, without the patience and support of whom I would never have been able to complete this project.
