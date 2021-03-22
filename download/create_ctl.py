import sys

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

if __name__ == '__main__':
	feature = sys.argv[1]
	year = int(sys.argv[2])
	if feature == 'rain':
		create_ctl_rain(year=year)
	elif feature == 'temp':
		create_ctl_temp(year=year)
	print(f"Created control files for {feature} {year}")
