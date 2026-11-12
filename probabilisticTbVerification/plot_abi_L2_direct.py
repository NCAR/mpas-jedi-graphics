import argparse
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import os, time, itertools
import logging
from netCDF4 import Dataset
import plot
import utils as ut
ut.logg()

'''
Standalone plot of ABI L2 cloud products (from NOAA CLASS) at their original resolution.
Expensive: tricontourf over every valid pixel of the full disk.
'''

# product -> variable to plot, and resolution of the lat/lon file
info = {'ACHAF': {'variable':    'HT', 'resol': '10km'},
        'ACHTF': {'variable':  'TEMP', 'resol':  '2km'},
        'ACMF' : {'variable':   'BCM', 'resol':  '2km'},
        'ACTPF': {'variable': 'Phase', 'resol':  '2km'},
        'CTPF' : {'variable':  'PRES', 'resol': '10km'},
       }

def read_and_plot(args):
    product, datestr, obsDir, latlonDir, outDir = args
    vdate = datetime.strptime(datestr, '%Y%m%d%H')
    savename = os.path.join(outDir, 'obs_'+product+'_v'+datestr)

    # find the scan starting at the top of the hour, e.g. OR_ABI-L2-ACHAF-M3_G16_s20181050600422_...
    productDir = os.path.join(obsDir, 'data_from_noaa_class_'+product)
    scanStart = 's'+vdate.strftime('%Y%j%H')+'00'
    files = [f for f in sorted(os.listdir(productDir)) if scanStart in f]
    if len(files) == 0:
      logging.warning('No %s file for %s in %s', product, datestr, productDir)
      return

    with Dataset(os.path.join(productDir, files[0]), 'r') as nf:
      obs = nf.variables[info[product]['variable']][:].flatten()
      qcf = nf.variables['DQF'][:].flatten()
    with Dataset(os.path.join(latlonDir, 'latlon_'+info[product]['resol']+'.nc'), 'r') as nf:
      lat = nf.variables['lat'][:].flatten()
      lon = nf.variables['lon'][:].flatten()

    # keep pixels with valid obs, valid lat/lon, and good quality (DQF == 0)
    valid = ~np.ma.getmaskarray(obs) & ~np.ma.getmaskarray(lat) & (np.ma.getdata(qcf) == 0)
    plot.field(np.ma.getdata(lon)[valid], np.ma.getdata(lat)[valid], np.ma.getdata(obs)[valid],
               product+' '+info[product]['variable']+' v'+datestr, savename, cmap='jet')

def main():
    h0 = time.time()
    logging.info('Starting '+__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument('-obs', '--obsDir', help='Directory containing data_from_noaa_class_<product>/', required=True)
    ap.add_argument('-latlon', '--latlonDir', help='Directory containing latlon_<resol>.nc', required=True)
    ap.add_argument('-p', '--products', type=ut.list_of_strings, default=','.join(info.keys()),
                    help='Comma-separated products from: '+','.join(info.keys()))
    ap.add_argument('-di', '--dateIni', help='Initial date (YYYYMMDDHH)', required=True)
    ap.add_argument('-de', '--dateEnd', help='Final date (YYYYMMDDHH)', required=True)
    ap.add_argument('-dt', '--delta', type=int, default=6, help='Interval between dates (hours)')
    ap.add_argument('-o', '--outDir', default='.', help='Output directory')
    ap.add_argument('-np', '--nprocs', type=int, default=1, help='Number of processes')
    args = ap.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    dateList, _ = ut.dateList(args.dateIni, args.dateEnd, args.delta)
    cases = [(p, d, args.obsDir, args.latlonDir, args.outDir) for p, d in itertools.product(args.products, dateList)]
    with Pool(processes=max(1, min(args.nprocs, len(cases))), maxtasksperchild=1) as pool:
      pool.map(read_and_plot, cases, chunksize=1)

    logging.info('Finished '+__name__+' successfully')
    logging.info('Time elapsed: {}'.format(str(time.time() - h0)))

if __name__ == '__main__':
    main()
