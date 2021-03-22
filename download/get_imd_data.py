import requests
from tqdm import tqdm
from cdo import Cdo
import os
import sys

cdo = Cdo()

def create_ctl_temp(year=1901):
    days = 366 if year % 4 == 0 else 365
    template = """DSET ./tmp.grd
TITLE 1 degree analyzed grids
UNDEF 99.9
XDEF 31 LINEAR 67.5 1
YDEF 31 LINEAR 7.5 1
ZDEF 1 Linear 1 1
TDEF {days} LINEAR 1JAN{year} 1DY
VARS 1
T 0 99 DAILYTEMP
ENDVARS""".format(year=year, days=days)
    with open('tmp.ctl', 'w') as ctl:
        ctl.write(template)

def create_ctl_rain(year=1901):
    days = 366 if year % 4 == 0 else 365
    template = """DSET ./tmp.grd
TITLE 0.25 degr analyzed grids
UNDEF -999.0
XDEF 135 LINEAR 66.5 0.25
YDEF 129 LINEAR 6.5 0.25
ZDEF 1 linear 1 1
* CHANGE TDEF TO 366 FOR LEAP YEARS
TDEF {days} LINEAR 1jan{year} 1DY
VARS 1
rf 0 99 GRIDDED RAINFALL
ENDVARS""".format(year=year, days=days)
    with open('tmp.ctl', 'w') as ctl:
        ctl.write(template)

def main(variable, feature, year):
    url = f"https://www.imdpune.gov.in/Clim_Pred_LRF_New/{feature}.php"
    # Streaming, so we can iterate over the response.
    response = requests.post(url, stream=True, data={feature: year})
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open('tmp.grd', 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if variable == 'rain':
        create_ctl_rain(year=year)
    else:
        create_ctl_temp(year=year)
    cdo.import_binary(input = 'tmp.grd', output='{feature}-{year}.nc')
    os.system('rm tmp.*')


if __name__ == '__main__':
    variable = sys.argv[1]
    feature = sys.argv[2]
    year = int(sys.argv[3])
    main(variable=variable, feature=feature, year=year)
