#!/usr/bin/env python3

import basic_plot_functions as bpf
import binning_utils as bu
from copy import deepcopy
import numpy as np
import var_utils as vu

from analysis.AnalysisBase import AnalysisBase
import analysis.StatisticsDatabase as sdb

class MultiDimBinMethodBase(AnalysisBase):
    '''
    Base class used to analyze statistics across binMethods with numerical binValues
      that are assigned numerical values, e.g., altitude, pressure, latitude, cloud fraction
    '''

    parallelism = True
    maxBinVarTier = 2

    cldFracTransform = 'logit'

    # name treated as the unfiltered/default vertical-level-filter variant; never
    # appears in output filenames/titles (see binning_utils.py's verticalBinFilters
    # and binFilterFile() below)
    blankBinFilterFile = bu.blankBinFilterFile

    def __init__(self, db:sdb, analysisType:str, diagnosticGroupings:dict):
        super().__init__(db, analysisType, diagnosticGroupings)
        # default 1D binVars
        # NOTE: vertical-level filtering (binFilters minvalue/maxvalue, possibly multiple named
        # ranges) for obsVarAlt, obsVarImpact, obsVarPrs, modVarDiagPrs, and modVarLev below is
        # configured in binning_utils.py's verticalBinFilters dict -- edit that dict, not
        # this source file, to change it.
        self.binVarDict = {
            vu.obsVarAlt: {
              'profilefunc': bpf.plotProfile,
              'binFilters': bu.verticalRanges(vu.obsVarAlt),
            },
            vu.obsVarImpact: {
              'profilefunc': bpf.plotProfile,
              'binFilters': bu.verticalRanges(vu.obsVarImpact),
            },
            vu.obsVarACI: {'profilefunc': bpf.plotSeries, 'binVarTier': 3},
            vu.obsVarCldFracX: {'profilefunc': bpf.plotSeries, 'binVarTier': 2},
            vu.obsVarCldFracY: {'profilefunc': bpf.plotSeries, 'binVarTier': 1},
            vu.obsVarLat: {'profilefunc': bpf.plotProfile},
            vu.obsVarPrs: {
              'profilefunc': bpf.plotProfile,
              'binFilters': bu.verticalRanges(vu.obsVarPrs),
            },
            vu.obsVarCI: {'profilefunc': bpf.plotSeries, 'binVarTier': 2},
            vu.obsVarLogCI: {'profilefunc': bpf.plotSeries, 'binVarTier': 3},
            vu.modVarDiagPrs: {
              'profilefunc': bpf.plotProfile,
              'binFilters': bu.verticalRanges(vu.modVarDiagPrs),
            },
            # vu.modVarLat is redundant with vu.obsVarLat (both have varShort=="lat")
            #vu.modVarLat: {'profilefunc': bpf.plotProfile, 'binVarTier': 1},
            vu.modVarLev: {
              'profilefunc': bpf.plotProfile,
              'binFilters': bu.verticalRanges(vu.modVarLev),
            },
            vu.obsVarGlint: {'profilefunc': bpf.plotSeries, 'binVarTier': 3},
            vu.obsVarLandFrac: {'profilefunc': bpf.plotSeries, 'binVarTier': 3},
            vu.obsVarLT: {'profilefunc': bpf.plotSeries, 'binVarTier': 3},
            vu.obsVarSenZen: {'profilefunc': bpf.plotSeries, 'binVarTier': 3},
        }
        self.maxDiagnosticsPerAnalysis = 10 // self.nExp

    def analyze_(self, workers = None):
        useWorkers = (not self.blocking and self.parallelism and workers is not None)
        diagnosticGrouped = {}
        for diag in self.availableDiagnostics:
            diagnosticGrouped[diag] = False

        diagnosticGroupings = deepcopy(self.diagnosticGroupings)
        for group in list(diagnosticGroupings.keys()):
            diags = diagnosticGroupings[group]
            if (len(diags) > self.maxDiagnosticsPerAnalysis or
               not set(diags).issubset(set(list(self.availableDiagnostics)))):
                del diagnosticGroupings[group]
                continue
            for diag in diags: diagnosticGrouped[diag] = True

        for diag in self.availableDiagnostics:
            if not diagnosticGrouped[diag]:
                diagnosticGroupings[diag] = [diag]

        for diagnosticGroup, diagnosticNames in diagnosticGroupings.items():
            if len(diagnosticNames) > self.maxDiagnosticsPerAnalysis: continue
            if len(set(diagnosticNames) & set(self.availableDiagnostics)) == 0: continue
            diagnosticConfigs = {}
            selectedStatistics = set([])
            for diagnosticName in diagnosticNames:
                diagnosticConfigs[diagnosticName] = deepcopy(self.diagnosticConfigs[diagnosticName])
                selectedStatistics = set(list(selectedStatistics) +
                                         diagnosticConfigs[diagnosticName]['selectedStatistics'])
            availableStatistics = set([])
            for diagnosticName in diagnosticNames:
                diagnosticConfigs[diagnosticName] = deepcopy(self.diagnosticConfigs[diagnosticName])
                availableStatistics = set(list(availableStatistics) +
                                         diagnosticConfigs[diagnosticName]['availableStatistics'])
            if not set(self.requiredStatistics).issubset(availableStatistics): continue

            diagBinVars = self.db.dfw.levels('binVar', {'diagName': diagnosticNames})

            for fullBinVar, options in self.binVarDict.items():
                if options.get('binVarTier', 1) > self.maxBinVarTier: continue
                binVar = vu.varDictAll.get(fullBinVar, [None, fullBinVar])[1]
                if (binVar not in diagBinVars): continue
                binVarLoc = {}
                binVarLoc['diagName'] = diagnosticNames
                binVarLoc['binVar'] = binVar
                binVarLoc['binVal'] = self.allBinNumVals2DasStr

                #Make figures for all binMethods
                binMethods = self.db.dfw.levels('binMethod', binVarLoc)
                binFilters = options.get('binFilters', {self.blankBinFilterFile: {}})
                for binMethod in binMethods:

                    #TODO: REMOVE, for testing only
                    #if binMethod != bu.identityBinMethod: continue

                    #Make figures for all named vertical-level filters (usually just
                    #'full', i.e. unfiltered, unless additional named ranges are configured
                    #in binning_utils.py's verticalBinFilters)
                    for filterName, binFilter in binFilters.items():
                        self.logger.info(diagnosticGroup+', '+binVar+', '+binMethod+', '+filterName)

                        if useWorkers:
                            workers.apply_async(self.innerloopsWrapper,
                                args = (diagnosticGroup, diagnosticConfigs, binVar, binMethod, selectedStatistics, options, filterName, binFilter))
                        else:
                            self.innerloopsWrapper(
                                diagnosticGroup, diagnosticConfigs, binVar, binMethod, selectedStatistics, options, filterName, binFilter)

    @staticmethod
    def maskByBinFilter(numVals, binFilter):
        '''
        Given an array-like of numeric binVals and a binFilter dict with optional
        'minvalue'/'maxvalue' keys, return a boolean mask that is True where the
        value should be REMOVED (i.e., keep numVals[~mask]). Used to implement
        vertical-level filtering (see binning_utils.py's verticalBinFilters)
        for both single-axis (MultiDimBinMethodBase) and 2D (BinValAxes2D) plots.
        Thin wrapper around bu.maskByRange.
        '''
        return bu.maskByRange(numVals, binFilter)

    def binFilterFile(self, filterName):
        '''
        Format a vertical-level filter name for file/title naming, mirroring
        binMethodFile(): the default/unfiltered name (blankBinFilterFile) is left off
        entirely; any other name becomes a distinguishing '_'+filterName suffix.
        '''
        if filterName is None or filterName == self.blankBinFilterFile:
            return ''
        return '_'+filterName

    def innerloopsWrapper(self,
        diagnosticGroup, diagnosticConfigs, binVar, binMethod, selectedStatistics, options,
        filterName, binFilter):

        myLoc = {}
        myLoc['binVar'] = binVar
        myLoc['binVal'] = self.allBinNumVals2DasStr
        myLoc['binMethod'] = binMethod

        # narrow mydfwDict by binVar, binVal, and binMethod to reduce run-time and memory
        mydfwDict = {'dfw': self.db.loc(myLoc)}

        # aggregate statistics when requested
        if self.requestAggDFW:
            mydfwDict['agg'] = sdb.DFWrapper.fromAggStats(mydfwDict['dfw'], ['cyDTime'])
            sdb.createORreplaceDerivedDiagnostics(mydfwDict['agg'], diagnosticConfigs)

        # further narrow mydfwDict by diagName
        # NOTE: derived diagnostics may require multiple diagName values;
        # can only narrow by diagName after aggregation
        myLoc['diagName'] = list(diagnosticConfigs.keys())
        for key in mydfwDict.keys():
            mydfwDict[key] = sdb.DFWrapper.fromLoc(mydfwDict[key], myLoc)

        # not a database query field; only used downstream (e.g. filenames/titles) via myLoc
        myLoc['binFilterName'] = filterName

        ## Get all float/int binVals associated with binVar
        binStrVals = mydfwDict['dfw'].levels('binVal')
        binUnits = mydfwDict['dfw'].uniquevals('binUnits')[0]

        # assume all bins represent same variable/units
        binLabel = binVar
        if binUnits != vu.miss_s:
            binLabel += ' ('+binUnits+')'
        for orig, sub in self.labelReplacements.items():
            binLabel = binLabel.replace(orig, sub)

        # bin info
        binNumVals = []
        for binVal in binStrVals:
            ibin = self.allBinStrVals.index(binVal)
            binNumVals.append(self.allBinNumVals[ibin])

        # filter out binVals less than (greater than) minvalue (maxvalue)
        # works for any vertical coordinate type (model level, pressure, height, ...)
        # since binFilter is scoped per binVar in binVarDict; set minvalue==maxvalue
        # to keep a single level, or omit one bound to only clip the other end.
        # For vertical binVars, minvalue/maxvalue are set via binning_utils.py's
        # verticalBinFilters dict (see __init__ above), not hardcoded here. The specific
        # named filter to apply (usually just 'full', i.e. unfiltered) is selected by
        # analyze_() and passed in as binFilter/filterName above.
        if binFilter:
          remove = self.maskByBinFilter(binNumVals, binFilter)
          binStrVals = list(np.asarray(binStrVals)[~remove])
          binNumVals = list(np.asarray(binNumVals)[~remove])

        # special independent variable axis configs
        binVarIs = {}
        specialBinVars = [
          vu.obsVarPrs,
          vu.obsVarMCI,
          vu.obsVarOCI,
          vu.obsVarLogCI,
          vu.obsVarCldFracX,
          vu.obsVarCldFracY,
          vu.modVarDiagPrs,
        ]
        for var in specialBinVars:
            var_dict = vu.varDictAll.get(var,['',''])
            binVarIs[var] = (var_dict[1] == binVar)

        binConfig = deepcopy(bpf.defaultIndepConfig)
        pCoord = binVarIs[vu.obsVarPrs] or binVarIs[vu.modVarDiagPrs]
        binConfig['invert'] = pCoord
        if pCoord: binConfig['transform'] = 'Pressure'
        if binVarIs[vu.obsVarMCI] or binVarIs[vu.obsVarOCI] or binVarIs[vu.obsVarLogCI]:
            binConfig['transform'] = 'CloudImpact'
        if binVarIs[vu.obsVarCldFracX] or binVarIs[vu.obsVarCldFracY]:
            binConfig['transform'] = self.cldFracTransform

        # sort bins by numeric value
        indices = list(range(len(binNumVals)))
        indices.sort(key=binNumVals.__getitem__)
        binNumVals = list(map(binNumVals.__getitem__, indices))
        binStrVals = list(map(binStrVals.__getitem__, indices))

        myBinConfigs = {
            'str': binStrVals,
            'num': binNumVals,
            'binLabel': binLabel,
            'binConfig': binConfig,
        }
        if len(binStrVals) < 2: return

        # only analyze variables that have non-zero Count when sliced by myLoc
        nVarsLoc = 0
        varMapLoc = []
        for (varName, varLabel) in self.varMap:
          if 'Count' in selectedStatistics:
              countDF = mydfwDict['dfw'].loc({'varName': varName}, 'Count')
              if countDF.shape[0] > 0:
                if np.nansum(countDF.to_numpy()) > 0:
                  nVarsLoc += 1
                  varMapLoc.append((varName, varLabel))
          else:
              statDF = mydfwDict['dfw'].loc({'varName': varName}, list(selectedStatistics)[0])
              if statDF.shape[0] > 0:
                if np.isfinite(statDF.to_numpy()).sum() > 0:
                  nVarsLoc += 1
                  varMapLoc.append((varName, varLabel))

        for statName in selectedStatistics:
            if statName not in options.get('onlyStatNames', selectedStatistics): continue

            self.innerloops(
                mydfwDict, diagnosticGroup, myLoc, statName, nVarsLoc, varMapLoc, myBinConfigs, options)

    def innerloops(self,
        dfwDict, diagnosticGroup, myLoc, statName, nVarsLoc, varMapLoc, myBinConfigs, options):
        '''
        virtual method
        '''
        raise NotImplementedError()
