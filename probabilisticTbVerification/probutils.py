import numpy as np
import warnings

# numpy >= 2.0 renamed trapz to trapezoid
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

tableCountKeys = ['obsYes', 'obsNo']

############################# THRESHOLDS ######################################

def parse_threshold(threshold):
    '''
    'pct_5.0'   -> ('pct', 5.0)   : percentile of each field (obs, each member separately)
    'val_235.0' -> ('val', 235.0) : fixed physical value (K)
    '''
    method, value = threshold.split('_')
    if method not in ('pct', 'val'):
      raise ValueError('threshold must look like pct_<percentile> or val_<value>: '+threshold)
    return method, float(value)

def threshold_tag(threshold):
    method, value = parse_threshold(threshold)
    return method+'thres_'+'{:05.1f}'.format(value)

def physical_threshold(threshold, field):
    method, value = parse_threshold(threshold)
    if method == 'pct':
      return float(np.percentile(field, value))
    return value

def event(field, value, direction):
    if direction == 'below':
      return field < value
    elif direction == 'above':
      return field > value
    else:
      raise NotImplementedError()

############################# CONTINGENCY TABLE ###############################

def prob_bin_index(nYes, nMem, nBins):
    # bin 0 is [0, 1/nBins], bin i>0 is (i/nBins, (i+1)/nBins]
    # integer arithmetic on member counts avoids round-off at the bin edges
    idx = (nYes*nBins + nMem - 1)//nMem - 1
    return np.clip(idx, 0, nBins-1)

def contingency_table(nYes, nMem, obsEvent, nBins):
    '''
    Nx2 probabilistic contingency table
      nYes     : number of members forecasting the event at each location
      obsEvent : boolean observed event at each location
    '''
    idx = prob_bin_index(nYes, nMem, nBins)
    return {
      'nBins': int(nBins),
      'nMembers': int(nMem),
      'nCases': 1,
      'obsYes': np.bincount(idx[obsEvent], minlength=nBins).tolist(),
      'obsNo': np.bincount(idx[~obsEvent], minlength=nBins).tolist(),
    }

def aggregate_tables(tables):
    tables = [t for t in tables if t is not None]
    if len(tables) == 0:
      return None
    agg = {k: tables[0][k] for k in ['nBins', 'nMembers']}
    for t in tables[1:]:
      if t['nBins'] != agg['nBins'] or t['nMembers'] != agg['nMembers']:
        raise ValueError('cannot aggregate tables with different nBins or nMembers')
    agg['nCases'] = int(sum(t['nCases'] for t in tables))
    for k in tableCountKeys:
      agg[k] = np.sum([t[k] for t in tables], axis=0).tolist()
    return agg

def bin_edges(nBins):
    return np.linspace(0., 1., nBins+1)

def bin_centers(nBins):
    edges = bin_edges(nBins)
    return 0.5*(edges[:-1] + edges[1:])

############################# SCORES ##########################################

def safe_divide(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
      return np.where(b > 0, np.asarray(a, dtype=float)/b, np.nan)

def reliability(table):
    '''
    probability bin center, observed relative frequency, and sample count per bin
      As in MET, the forecast probability of each bin is its center (midpoint of the bin edges),
      not the mean forecast probability of the samples in the bin.
    '''
    yes = np.array(table['obsYes'], dtype=float)
    no = np.array(table['obsNo'], dtype=float)
    count = yes + no
    obsFreq = safe_divide(yes, count)
    return bin_centers(table['nBins']), obsFreq, count

def roc(table):
    '''
    POFD, POD at each probability bin edge (forecast yes when probability is above the edge),
    from (1,1) to (0,0), and the area under the curve
    '''
    yes = np.array(table['obsYes'], dtype=float)
    no = np.array(table['obsNo'], dtype=float)
    # counts at or above each bin, with a trailing 0 for the "never yes" point
    yesAbove = np.append(np.cumsum(yes[::-1])[::-1], 0.)
    noAbove = np.append(np.cumsum(no[::-1])[::-1], 0.)
    pod = safe_divide(yesAbove, yes.sum())
    pofd = safe_divide(noAbove, no.sum())
    if np.all(np.isfinite(pod)) and np.all(np.isfinite(pofd)):
      auc = float(-_trapezoid(pod, pofd)) # pofd decreases along the curve
    else:
      auc = np.nan
    return pofd, pod, auc

def brier(table):
    '''
    Brier score (BS), sample climatology BS, and Brier skill score (BSS)
      BS = sum_i [ obsYes_i*(1-p_i)^2 + obsNo_i*p_i^2 ] / N, with p_i the center of probability bin i
      This follows MET, which computes BS from the Nx2 contingency table at the bin centers:
        https://metplus.readthedocs.io/projects/met/en/latest/Users_Guide/appendixC.html#brier-score
        https://github.com/dtcenter/MET/blob/develop/src/libcode/vx_statistics/contable_nx2.cc
      It differs slightly from the BS of the individual forecast probabilities.
      BS_clim = obar*(1-obar), with obar the observed event frequency of the sample
      Reference forecast: sample climatology, i.e., the constant probability obar everywhere.
        - obar is computed from the same table (all initialization times and locations for one
          lead time and threshold), so BS_clim equals the uncertainty term of the BS decomposition.
        - pct_<p> thresholds: obar = p/100 by construction (up to removed locations), so
          BS_clim = (p/100)*(1-p/100).
        - val_<v> thresholds: obar depends on the observations, so it can change with lead time
          (different valid times). BSS then measures skill relative to each sample's own base rate.
    '''
    yes = np.array(table['obsYes'], dtype=float)
    no = np.array(table['obsNo'], dtype=float)
    total = yes.sum() + no.sum()
    if total == 0:
      return np.nan, np.nan, np.nan
    p = bin_centers(table['nBins'])
    bs = np.sum(yes*(1. - p)**2 + no*p**2)/total
    obar = yes.sum()/total
    bsClim = obar*(1. - obar)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', category=RuntimeWarning)
      bss = 1. - bs/bsClim if bsClim > 0. else np.nan
    return bs, bsClim, bss
