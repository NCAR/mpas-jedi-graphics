import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs

dpi = 200

def _map_axes(ax, extent, labels=True):
    ax.coastlines()
    if extent is None:
      ax.set_global()
    else:
      ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=labels, xlocs=np.arange(-180, 181, 60), linestyle='--')

def field(lons, lats, data, title, savename, vmin=None, vmax=None, cmap='viridis', extent=None):
    if not (lons.shape[0] == lats.shape[0] == data.shape[0]):
      raise ValueError('inconsistent lon/lat/data dimensions for '+savename)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    _map_axes(ax, extent)
    levels = np.linspace(vmin, vmax, 31) if (vmin is not None and vmax is not None) else 30
    cont = ax.tricontourf(lons, lats, data.astype(float), levels,
                          extend='both', transform=ccrs.PlateCarree(), cmap=cmap)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.2, axes_class=plt.Axes)
    ticks = np.linspace(vmin, vmax, 7) if (vmin is not None and vmax is not None) else None
    plt.colorbar(cont, cax=cax, orientation='horizontal', ticks=ticks)
    plt.savefig(savename+'.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def ensemble_fields(lons, lats, data, title, savename, vmin=None, vmax=None, cmap='jet', extent=None):
    '''
    one small map per member; data is (nlocs, nMembers)
    '''
    nMem = data.shape[1]
    ncols = min(nMem, 14)
    nrows = int(np.ceil(nMem/ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 1.0*nrows+1), layout='constrained',
                            subplot_kw={'projection': ccrs.PlateCarree()}, squeeze=False)
    levels = np.linspace(vmin, vmax, 31) if (vmin is not None and vmax is not None) else 30
    for ie, ax in enumerate(axs.flat):
      if ie >= nMem:
        ax.set_visible(False)
        continue
      _map_axes(ax, extent, labels=False)
      ax.tricontourf(lons, lats, data[:, ie], levels, extend='both',
                     transform=ccrs.PlateCarree(), cmap=cmap)
      ax.set_title('mem '+str(ie+1), fontsize=5)
    fig.suptitle(title)
    plt.savefig(savename+'.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def histogram(obs, ens, title, savename, bins=np.arange(180, 321, 4)):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist([obs, ens.flatten()], bins=bins, density=True,
            label=['obs ({:6.2f})'.format(np.mean(obs)), 'ensemble hofx ({:6.2f})'.format(np.mean(ens))])
    ax.legend(loc='upper right')
    ax.set_title(title)
    ax.set_xlabel('Brightness Temperature [K]')
    ax.set_ylabel('Relative frequency')
    plt.savefig(savename+'.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def reliability(curves, title, savename):
    '''
    curves: list of (prob, obsFreq, count, label)
    the inset shows the sample count in each probability bin (log scale)
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k:', label='Perfect')
    axin = ax.inset_axes([0.08, 0.62, 0.3, 0.3])
    for prob, obsFreq, count, label in curves:
      filled = count > 0 # skip empty bins so the line is not broken
      line, = ax.plot(prob[filled], obsFreq[filled], 'o-', label=label)
      axin.plot(prob[filled], count[filled], '.-', color=line.get_color())
    axin.set_yscale('log')
    axin.set_title('Sample count', fontsize='small')
    axin.tick_params(labelsize='x-small')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Observed Relative Frequency')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize='small')
    ax.grid(True)
    plt.savefig(savename+'.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def roc(curves, title, savename):
    '''
    curves: list of (pofd, pod, auc, label)
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k:', label='No skill')
    for pofd, pod, auc, label in curves:
      ax.plot(pofd, pod, 'o-', label='{} (AUC={:.4f})'.format(label, auc))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('POFD')
    ax.set_ylabel('POD')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize='small')
    ax.grid(True)
    plt.savefig(savename+'.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def lead_time(hours, lines, ylabel, title, savename):
    '''
    lines: dict of label -> values at each forecast hour
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, values in lines.items():
      ax.plot(hours, values, 'o-', label=label)
    ax.set_xticks(list(hours))
    ax.set_xlabel('Forecast lead time [h]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize='small')
    ax.grid(True)
    plt.savefig(savename+'.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
