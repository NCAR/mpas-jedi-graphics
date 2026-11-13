import argparse
from datetime import datetime, timedelta
from multiprocessing import Pool
import numpy as np
import os, time, itertools
import logging
import yaml
from netCDF4 import Dataset
import plot
import probutils as pu
import utils as ut
ut.logg()

defaultTemplate = '{expDir}/Verification/fc/mem{mem:03d}/{init}/{fhr}hr/dbOut/obsout_{app}_{sensor}.h5'
tbRange = (190., 310.)

class ProbabilisticVerification:
  '''
  Probabilistic verification of ensemble brightness temperature (Tb) forecasts in observation space.

  For each forecast initialization time, lead time, and threshold:
  - an event is defined at each location as Tb below (or above) the threshold, for the obs and for each member
  - the forecast probability is the fraction of members forecasting the event
  - an Nx2 contingency table (probability bin x observed yes/no) is written out
  The tables are aggregated over initialization times, and reliability diagrams, ROC curves,
  and Brier (skill) scores are computed from the aggregated tables.

  Thresholds:
    pct_<p>: p-th percentile, computed separately for the obs and for each member, so the
             obs and each member have the same event frequency (systematic bias is removed)
    val_<v>: fixed value in K
  '''
  def __init__(self, args):
      self.name = 'ProbabilisticVerification'
      logging.info(self.name)

      self.expDir = args.expDir
      self.template = args.fileTemplate
      self.app = args.app
      self.sensor = args.sensor
      self.channel = args.channel
      self.nMem = args.nMembers
      self.thresholds = args.thresholds
      for t in self.thresholds:
        pu.parse_threshold(t) # fail early on a bad threshold
      self.nBins = args.nBins
      self.eventDirection = args.event
      self.extent = args.extent
      self.plotMaps = args.plotMaps
      self.plotEnsemble = args.plotEnsemble
      self.plotHistogram = args.histogram
      self.plotCases = args.plotCases

      self.Nhours = list(range(args.firstHour, args.forecastHours+1, args.fcstFrequency))
      self.dateList, _ = ut.dateList(args.dateIni, args.dateEnd, args.delta)

      self.tableDir = os.path.join(args.outDir, 'tables')
      self.mapDir = os.path.join(args.outDir, 'maps')
      self.scoreDir = os.path.join(args.outDir, 'scores')
      for d in [self.tableDir, self.mapDir, self.scoreDir]:
        os.makedirs(d, exist_ok=True)

      self.workers = args.nprocs if args.nprocs > 0 else len(os.sched_getaffinity(0))
      self.readers = args.nreaders
      logging.info('Number of cpus available: {}'.format(str(self.workers)))


  def driver(self, plotOnly=False):

      if plotOnly:
        logging.info('Reading aggregated tables from %s', self.tableDir)
        aggTables = self.read_aggregated_tables()
      else:
        if self.nMem <= 0:
          self.nMem = self.count_members()
        logging.info('Number of members: %d', self.nMem)

        cases = list(itertools.product(self.dateList, self.Nhours))
        results = self.read_and_process(cases)

        logging.info('Aggregating contingency tables')
        aggTables = {}
        for hr, thres in itertools.product(self.Nhours, self.thresholds):
          tables = [r[thres] for (d, h), r in zip(cases, results) if h == hr and r is not None]
          aggTables[(hr, thres)] = pu.aggregate_tables(tables)
          self.write_table(aggTables[(hr, thres)], self.aggregated_table_name(hr, thres))

      logging.info('Making score plots')
      self.make_score_plots(aggTables)


    ###################### END OF DRIVER #########################


  def file_name(self, mem, datestr, hr):
      return self.template.format(expDir=self.expDir, mem=mem, init=datestr, fhr=hr, app=self.app, sensor=self.sensor)

  def count_members(self):
      # count members of the first available case
      for datestr, hr in itertools.product(self.dateList, self.Nhours):
        n = 0
        while os.path.exists(self.file_name(n+1, datestr, hr)):
          n += 1
        if n > 0:
          return n
      raise FileNotFoundError('no member files found for '+self.file_name(1, self.dateList[0], self.Nhours[0]))

  def case_tag(self, datestr, hr):
      vdate = datetime.strptime(datestr, '%Y%m%d%H') + timedelta(hours=hr)
      return 'i'+datestr+'_f'+str(hr).zfill(3)+'_v'+vdate.strftime('%Y%m%d%H')

  def case_table_name(self, datestr, hr, thres):
      return os.path.join(self.tableDir, 'contingency_table_'+self.case_tag(datestr, hr)+'_'+pu.threshold_tag(thres)+'.yaml')

  def aggregated_table_name(self, hr, thres):
      return os.path.join(self.tableDir, 'contingency_table_aggregated_f'+str(hr).zfill(3)+'_'+pu.threshold_tag(thres)+'.yaml')

  def write_table(self, table, filename):
      if table is None:
        logging.warning('No data for %s', filename)
        return
      with open(filename, 'w') as f:
        yaml.safe_dump(table, f, sort_keys=False)

  def read_aggregated_tables(self):
      aggTables = {}
      for hr, thres in itertools.product(self.Nhours, self.thresholds):
        filename = self.aggregated_table_name(hr, thres)
        if os.path.exists(filename):
          with open(filename) as f:
            aggTables[(hr, thres)] = yaml.safe_load(f)
        else:
          logging.warning('Missing %s', filename)
          aggTables[(hr, thres)] = None
      return aggTables


  def read_and_process(self, cases):
      '''
      Reading the member files is I/O bound, so it is done by a pool of readers over
      (case, member), which overlaps the reads within and across cases. When all members of a
      case are read, the case is processed (tables, plots) by a second pool of workers.
      Pool workers cannot start their own pools, hence the two pools side by side.
      '''
      mpworkers = max(1, min(self.workers, len(cases)))
      mpreaders = max(1, min(self.readers, len(cases)*self.nMem))
      logging.info('Number of processes to be used: %d readers, %d workers', mpreaders, mpworkers)

      reads = [(case, im) for case in cases for im in range(self.nMem)]
      pending = {}
      with Pool(processes=mpreaders) as readPool, Pool(processes=mpworkers, maxtasksperchild=1) as casePool:
        members = []
        # imap keeps the order, so the members of each case arrive together
        for ((datestr, hr), im), member in zip(reads, readPool.imap(self.read_member, reads, chunksize=1)):
          members.append(member)
          if len(members) < self.nMem:
            continue
          case = (datestr, hr)
          data = self.assemble_case(datestr, hr, members)
          members = []
          if data is not None:
            pending[case] = casePool.apply_async(self.process_case, (case, data))
        results = [pending[case].get() if case in pending else None for case in cases]
      return results

  def read_member(self, args):
      '''
      returns hofx (nlocs) of one member, plus lats, lons, and obs for the first member,
      or None if the file is missing
      '''
      (datestr, hr), im = args
      filename = self.file_name(im+1, datestr, hr)
      if not os.path.exists(filename):
        logging.warning('Missing %s', filename)
        return None
      with Dataset(filename, 'r') as nf:
        ich = np.where(nf.variables['Channel'][:] == self.channel)[0]
        if len(ich) != 1:
          raise ValueError('channel {} not found in {}'.format(self.channel, filename))
        ich = ich[0]
        member = {'hofx': nf.groups['hofx'].variables['brightnessTemperature'][:, ich]}
        if im == 0:
          member['lats'] = nf.groups['MetaData'].variables['latitude'][:]
          member['lons'] = nf.groups['MetaData'].variables['longitude'][:]
          member['obs'] = nf.groups['ObsValue'].variables['brightnessTemperature'][:, ich]
      return member

  def assemble_case(self, datestr, hr, members):
      '''
      returns lats, lons, obs (nlocs), and ensemble hofx (nlocs, nMem) at locations valid for
      the obs and for all members, or None if a member file is missing
      '''
      if any(m is None for m in members):
        logging.warning('Skipping %s %dhr: missing member files', datestr, hr)
        return None
      lats, lons, obs = members[0]['lats'], members[0]['lons'], members[0]['obs']
      enshofx = np.ma.empty((obs.shape[0], self.nMem))
      for im, m in enumerate(members):
        enshofx[:, im] = m['hofx']

      valid = ~np.ma.getmaskarray(obs) & np.all(~np.ma.getmaskarray(enshofx), axis=1)
      nInvalid = np.count_nonzero(~valid)
      if nInvalid > 0:
        logging.info('%s %dhr: %d of %d locations removed (missing obs or hofx)', datestr, hr, nInvalid, len(valid))
      return (np.ma.getdata(lats)[valid], np.ma.getdata(lons)[valid],
              np.ma.getdata(obs)[valid], np.ma.getdata(enshofx)[valid, :])


  def process_case(self, case, data):
      datestr, hr = case
      logging.info('Working on: %s %d', datestr, hr)
      lats, lons, obs, enshofx = data
      tag = self.case_tag(datestr, hr)
      chname = 'ch'+str(self.channel)

      if self.plotMaps:
        plot.field(lons, lats, obs, 'obs Tb '+chname+' '+tag, os.path.join(self.mapDir, 'obs_Tb_'+chname+'_'+tag),
                   *tbRange, cmap='jet', extent=self.extent)
        plot.field(lons, lats, enshofx.mean(axis=1), 'ensemble mean Tb '+chname+' '+tag,
                   os.path.join(self.mapDir, 'ensmean_Tb_'+chname+'_'+tag), *tbRange, cmap='jet', extent=self.extent)
        plot.field(lons, lats, enshofx.std(axis=1), 'ensemble spread Tb '+chname+' '+tag,
                   os.path.join(self.mapDir, 'ensspread_Tb_'+chname+'_'+tag), cmap='viridis', extent=self.extent)
      if self.plotEnsemble:
        plot.ensemble_fields(lons, lats, enshofx, 'hofx Tb '+chname+' '+tag,
                             os.path.join(self.mapDir, 'enshofx_Tb_'+chname+'_'+tag), *tbRange, extent=self.extent)
      if self.plotHistogram:
        plot.histogram(obs, enshofx, 'Tb '+chname+' '+tag, os.path.join(self.mapDir, 'histogram_'+chname+'_'+tag))

      tables = {}
      for thres in self.thresholds:
        obsThres = pu.physical_threshold(thres, obs)
        memThres = [pu.physical_threshold(thres, enshofx[:, im]) for im in range(self.nMem)]
        obsEvent = pu.event(obs, obsThres, self.eventDirection)
        nYes = np.sum([pu.event(enshofx[:, im], memThres[im], self.eventDirection) for im in range(self.nMem)], axis=0)

        table = pu.contingency_table(nYes, self.nMem, obsEvent, self.nBins)
        table['physicalThreshold'] = {'obs': float(obsThres), 'members': [float(t) for t in memThres]}
        self.write_table(table, self.case_table_name(datestr, hr, thres))
        tables[thres] = table

        ttag = tag+'_'+pu.threshold_tag(thres)
        if self.plotMaps:
          plot.field(lons, lats, obsEvent.astype(float), 'obs event '+chname+' '+thres+' '+tag,
                     os.path.join(self.mapDir, 'obs_event_'+chname+'_'+ttag), 0., 1., cmap='Greens', extent=self.extent)
          plot.field(lons, lats, nYes/self.nMem, 'forecast probability '+chname+' '+thres+' '+tag,
                     os.path.join(self.mapDir, 'probability_'+chname+'_'+ttag), 0., 1., cmap='Greens', extent=self.extent)
        if self.plotCases:
          self.plot_scores([(table, '')], thres+' '+tag, ttag)

      return tables


  def plot_scores(self, tableLabels, title, savetag):
      tableLabels = [(t, l) for t, l in tableLabels if t is not None]
      if len(tableLabels) == 0:
        return
      plot.reliability([(*pu.reliability(t), l) for t, l in tableLabels],
                       'Reliability '+title, os.path.join(self.scoreDir, 'reliability_'+savetag))
      plot.roc([(*pu.roc(t), l) for t, l in tableLabels],
               'ROC '+title, os.path.join(self.scoreDir, 'roc_'+savetag))

  def make_score_plots(self, aggTables):
      chname = 'ch'+str(self.channel)

      # one plot per lead time and threshold
      for hr, thres in itertools.product(self.Nhours, self.thresholds):
        table = aggTables[(hr, thres)]
        if table is None:
          continue
        label = '{} cases'.format(table['nCases'])
        self.plot_scores([(table, label)], chname+' '+thres+' f'+str(hr).zfill(3),
                         'aggregated_'+chname+'_f'+str(hr).zfill(3)+'_'+pu.threshold_tag(thres))

      # all thresholds for each lead time
      for hr in self.Nhours:
        self.plot_scores([(aggTables[(hr, t)], t) for t in self.thresholds],
                         chname+' f'+str(hr).zfill(3), 'aggregated_summary_'+chname+'_f'+str(hr).zfill(3))

      # all lead times for each threshold
      for thres in self.thresholds:
        self.plot_scores([(aggTables[(h, thres)], 'f'+str(h).zfill(3)) for h in self.Nhours],
                         chname+' '+thres, 'aggregated_summary_'+chname+'_'+pu.threshold_tag(thres))

      # BS, BSS, AUC as a function of lead time
      scores = {'BS': {}, 'BSS': {}, 'AUC': {}}
      for thres in self.thresholds:
        for s in scores:
          scores[s][thres] = []
        for hr in self.Nhours:
          table = aggTables[(hr, thres)]
          bs, bsClim, bss = pu.brier(table) if table is not None else (np.nan,)*3
          auc = pu.roc(table)[2] if table is not None else np.nan
          scores['BS'][thres].append(bs)
          scores['BSS'][thres].append(bss)
          scores['AUC'][thres].append(auc)
      for s, lines in scores.items():
        plot.lead_time(self.Nhours, lines, s, s+' '+chname, os.path.join(self.scoreDir, s+'_aggregated_summary_'+chname))
      with open(os.path.join(self.scoreDir, 'scores_aggregated_summary_'+chname+'.yaml'), 'w') as f:
        yaml.safe_dump({'forecastHours': self.Nhours,
                        **{s: {t: [float(v) for v in vals] for t, vals in lines.items()} for s, lines in scores.items()}},
                       f, sort_keys=False)


def main():
    h0 = time.time()
    logging.info('Starting '+__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument('-dir', '--expDir', help='Experiment directory (fills {expDir} in the file template)', required=True)
    ap.add_argument('-tmpl', '--fileTemplate', default=defaultTemplate,
                    help='Member obsout file template with {expDir}, {mem}, {init}, {fhr}, {app}, {sensor}. Default: '+defaultTemplate)
    ap.add_argument('-app', '--app', default='hofx', help='Application name in the obsout file name')
    ap.add_argument('-s', '--sensor', default='abi_g16', help='Sensor, e.g. abi_g16, ahi_himawari8')
    ap.add_argument('-ch', '--channel', type=int, default=13, help='Channel number (value of the Channel coordinate)')
    ap.add_argument('-nmem', '--nMembers', type=int, default=0, help='Number of members; 0 counts the existing member files')
    ap.add_argument('-di', '--dateIni', help='Initial forecast initialization date (YYYYMMDDHH)', required=True)
    ap.add_argument('-de', '--dateEnd', help='Final forecast initialization date (YYYYMMDDHH)', required=True)
    ap.add_argument('-dt', '--delta', type=int, default=24, help='Interval between initialization dates (hours)')
    ap.add_argument('-fhr0', '--firstHour', type=int, default=6, help='First forecast lead time (hours)')
    ap.add_argument('-fhr', '--forecastHours', type=int, default=48, help='Last forecast lead time (hours)')
    ap.add_argument('-freq', '--fcstFrequency', type=int, default=6, help='Interval between forecast lead times (hours)')
    ap.add_argument('-thres', '--thresholds', type=ut.list_of_strings, default='pct_5.0,pct_10.0,pct_15.0,pct_20.0,pct_25.0',
                    help='Comma-separated thresholds: pct_<percentile> and/or val_<Tb in K>')
    ap.add_argument('-ev', '--event', choices=['below', 'above'], default='below',
                    help='Event is Tb below (cold, cloudy) or above the threshold')
    ap.add_argument('-nbins', '--nBins', type=int, default=20, help='Number of forecast probability bins')
    ap.add_argument('-ext', '--extent', type=lambda s: [float(v) for v in s.split(',')], default=None,
                    help='Map extent lon0,lon1,lat0,lat1; global when not set')
    ap.add_argument('-maps', '--plotMaps', type=ut.str2bool, default=False, help='Plot obs, ensemble mean/spread, event, and probability maps')
    ap.add_argument('-ens', '--plotEnsemble', type=ut.str2bool, default=False, help='Plot maps of each member (expensive)')
    ap.add_argument('-hist', '--histogram', type=ut.str2bool, default=False, help='Plot obs and ensemble hofx Tb histograms')
    ap.add_argument('-cases', '--plotCases', type=ut.str2bool, default=False, help='Plot reliability and ROC for each case')
    ap.add_argument('-plotOnly', '--plotOnly', type=ut.str2bool, default=False, help='Only re-plot scores from existing aggregated tables')
    ap.add_argument('-o', '--outDir', default='.', help='Output directory')
    ap.add_argument('-np', '--nprocs', type=int, default=0, help='Number of processes for processing cases; 0 uses all available cpus')
    ap.add_argument('-nread', '--nreaders', type=int, default=16,
                    help='Number of processes reading member files; reading is I/O bound, so this can exceed the cpus')
    args = ap.parse_args()

    pv = ProbabilisticVerification(args)
    pv.driver(plotOnly=args.plotOnly)

    hf = time.time()
    ftime = hf - h0
    logging.info('Finished '+__name__+' successfully')
    logging.info('Time elapsed: {}'.format(str(ftime)))

if __name__ == '__main__':
    main()
