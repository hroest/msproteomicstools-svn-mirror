#!/usr/bin/env python
# -*- coding: utf-8  -*-
"""
=========================================================================
        msproteomicstools -- Mass Spectrometry Proteomics Tools
=========================================================================

Copyright (c) 2013, ETH Zurich
For a full list of authors, refer to the file AUTHORS.

This software is released under a three-clause BSD license:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of any author or any participating institution
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
--------------------------------------------------------------------------
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL ANY OF THE AUTHORS OR THE CONTRIBUTING
INSTITUTIONS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
--------------------------------------------------------------------------
$Maintainer: Hannes Roest$
$Authors: Hannes Roest$
--------------------------------------------------------------------------

python analysis/alignment/feature_alignment.py --in /media/data/tmp/smallset_sorted/*.csv --out /tmp/out_new.csv --readmethod minimal  --method LocalMSTAllCluster --max_rt_diff 75 --alignment_score 0.001 --realign_runs "WeightedNearestNeighbour"  --use_dscore_filter --dscore_cutoff -2.0 --out_ids /tmp/ids --max_fdr_quality 0.20

time python analysis/alignment/requantAlignedValues.py --peakgroups_infile /tmp/out_new2.csv  --out /tmp/imputed_new2.csv --do_single_run /media/data/tmp/strep_new4/hroest_K131126_023_SW_Spyo_M3_Musser_21.chrom.mzML --realign_runs "lowess" --method singleClosestRun  && diff /tmp/imputed_new2.csv imputed_expected.csv && echo "Success"

"""

import os, sys, csv, time
import numpy
import argparse
from msproteomicstoolslib.format.SWATHScoringReader import *
import msproteomicstoolslib.format.TransformationCollection as transformations
from msproteomicstoolslib.algorithms.alignment.SplineAligner import SplineAligner
from msproteomicstoolslib.algorithms.alignment.AlignmentMST import getDistanceMatrix
from msproteomicstoolslib.algorithms.PADS.MinimumSpanningTree import MinimumSpanningTree
from msproteomicstoolslib.algorithms.alignment.AlignmentHelper import write_out_matrix_file
import msproteomicstoolslib.algorithms.graphs.graphs as graphs
import msproteomicstoolslib.math.Smoothing as smoothing
from feature_alignment import Experiment

# The Window overlap which needs to be taken into account when calculating from which swath window to extract!
SWATH_EDGE_SHIFT = 1

class ImputeValuesHelper(object):
    """
    Static object with some helper methods.
    """

    @staticmethod
    def select_correct_swath(swath_chromatograms, mz):
        """Select the correct chromatogram

        Args:
            swath_chromatograms(dict): containing the objects pointing to the original chrom mzML (see run_impute_values)
            mz(float): the mz value of the precursor
        """
        mz = mz + SWATH_EDGE_SHIFT
        swath_window_low = int(mz / 25) *25
        swath_window_high = int(mz / 25) *25 + 25
        res = {}
        for k,v in swath_chromatograms.iteritems():
            # TODO smarter selection here
            selected = [vv for prec_mz,vv in v.iteritems() if prec_mz >= swath_window_low and prec_mz < swath_window_high]
            if len(v) == 1: 
                # We have merged chrom.mzML file (only one file)
                selected = v.values()
            if len(selected) == 1: 
                res[k] = selected[0]
        return res

    @staticmethod
    def convert_to_this(orig_runid, target_runid, ref_id, rt, transformation_collection_):
        """ Convert a retention time into one of the target RT space.
        
        Using the transformation collection
        """
        try:
            normalized_space_rt = transformation_collection_.getTransformation(orig_runid, ref_id).predict( [rt] )[0]
            return transformation_collection_.getTransformation(ref_id, target_runid).predict( [normalized_space_rt] )[0]
        except AttributeError as e:
            print "Could not convert from run %s to run %s (throug reference run %s)- are you sure you gave the correspoding trafo file with the --in parameter?" % (orig_runid, target_runid, ref_id)
            print e
            raise e


class SwathChromatogramRun(object):
    """ A single SWATH LC-MS/MS run.

    Each run may contain multiple files (split up by swath).
    """

    def __init__(self):
        self.chromfiles = []

    def parse(self, runid, files):
        """ Parse a set of files which all belong to the same experiment 
        """
        for i,f in enumerate(files):
            import pymzml
            if f.find("rtnorm") != -1 or f.find("ms1scan") != -1: 
                continue
            run = pymzml.run.Reader(f, build_index_from_scratch=True)
            if not run.info["seekable"]:
                raise Exception("Could not properly read file", f)
            self.chromfiles.append(run)

    def getChromatogram(self, chromid):
        for cfile in self.chromfiles:
            if cfile.info["offsets"].has_key(chromid):
                return cfile[chromid]
        return None

class SwathChromatogramCollection(object):
    """ A collection of multiple SWATH LC-MS/MS runs.

    Each single run is represented as a SwathChromatogramRun and accessible
    through a run id.
    """

    def __init__(self):
        self.allruns = {}
        self.cached_run = None
        self.cache = {}

    def createRunCache(self, runid):
        self.cache = {}
        self.cached_run = runid

        import copy
        for run in self.allruns[runid].chromfiles:
            for chromid, value in run.info['offsets'].iteritems():
              if value is None: continue
              self.cache[ chromid ] = copy.copy(run[chromid])

    def _getChromatogramCached(self, runid, chromid):
        assert runid == self.cached_run
        return self.cache.get( chromid, None)

    def getChromatogram(self, runid, chromid):
        if runid == self.cached_run:
            return self._getChromatogramCached(runid, chromid)

        if not self.allruns.has_key(runid):
            return None
        return self.allruns[runid].getChromatogram(chromid)

    def parse(self, trafo_fnames):
        """ Parse a set of different experiments.
        """
        swath_chromatograms = {}
        for filename in trafo_fnames:
            # get the run id
            start = time.time()
            f = open(filename, "r")
            header = f.next().split("\t")
            f.close()
            all_swathes = {}
            runid = header[1]
            import glob
            dname = os.path.dirname(filename)
            files = glob.glob(os.path.join(dname + "/*.mzML") )

            swathrun = SwathChromatogramRun()
            swathrun.parse(runid, files)
            self.allruns[runid] = swathrun
            print "Parsing chromatograms in", filename, "took %0.4fs" % (time.time() - start)

    def parseFromMzML(self, mzML_files, runIdMapping):
        """ Parse a set of different experiments.
        """
        swath_chromatograms = {}
        for filename in mzML_files:
            start = time.time()
            runid = runIdMapping[filename]
            swathrun = SwathChromatogramRun()
            swathrun.parse(runid, [filename])
            self.allruns[runid] = swathrun
            print "Parsing chromatograms in", filename, "took %0.4fs" % (time.time() - start)

    def get_runids(self):
        return self.allruns.keys()

def addDataToTrafo(tr_data, run_0, run_1, spl_aligner, multipeptides, realign_method, max_rt_diff):
    id_0 = run_0.get_id()
    id_1 = run_1.get_id()

    if id_0 == id_1:
        return

    # Data
    data_0, data_1 = spl_aligner._getRTData(run_0, run_1, multipeptides)
    tr_data.addData(id_0, data_0, id_1, data_1)

    # Smoothers
    sm_0_1 = smoothing.getSmoothingObj(options.realign_method, topN=3,
                                       max_rt_diff=max_rt_diff,
                                       min_rt_diff=0.1, removeOutliers=False,
                                       tmpdir=None)
    sm_1_0 = smoothing.getSmoothingObj(options.realign_method, topN=3,
                                       max_rt_diff=max_rt_diff,
                                       min_rt_diff=0.1, removeOutliers=False,
                                       tmpdir=None)

    # Add data
    sm_0_1.initialize(data_0, data_1)
    sm_1_0.initialize(data_1, data_0)
    tr_data.addTrafo(id_0, id_1, sm_0_1)
    tr_data.addTrafo(id_1, id_0, sm_1_0)

def runSingleFileImputation(options, peakgroups_file, mzML_file, method):
    """Impute values across chromatograms

    Args:
        peakgroups_file(filename): CSV file containing all peakgroups
        mzML_file(filename): mzML file containing chromatograms
    Returns:
        A tuple of:
            new_exp(AlignmentExperiment): experiment containing the aligned peakgroups
            multipeptides(list(AlignmentHelper.Multipeptide)): list of multipeptides

    This function will read the csv file with all peakgroups as well as the
    provided chromatogram file (.chrom.mzML). It will then try to impute
    missing values for those peakgroups where no values is currently present,
    reading the raw chromatograms.
    """

    fdr_cutoff_all_pg = 1.0
    start = time.time()
    reader = SWATHScoringReader.newReader([peakgroups_file], options.file_format, readmethod="complete")
    new_exp = Experiment()
    new_exp.runs = reader.parse_files()
    multipeptides = new_exp.get_all_multipeptides(fdr_cutoff_all_pg, verbose=False)
    print("Parsing the peakgroups file took %ss" % (time.time() - start) )

    mapping = {}
    inferMapping([ mzML_file ], [ peakgroups_file ], mapping, verbose=False)
    mapping_inv = dict([(v[0],k) for k,v in mapping.iteritems()])

    # Do only a single run : read only one single file
    start = time.time()
    swath_chromatograms = SwathChromatogramCollection()
    swath_chromatograms.parseFromMzML([ mzML_file ], mapping_inv)
    print("Reading the chromatogram files took %ss" % (time.time() - start) )
    assert len(swath_chromatograms.get_runids() ) == 1
    rid = swath_chromatograms.get_runids()[0]

    initial_alignment_cutoff = 0.0001
    max_rt_diff = 30
    tr_data = transformations.LightTransformationData()
    spl_aligner = SplineAligner(initial_alignment_cutoff)

    if method == "singleClosestRun":
        tree_mapped = None

        dist_matrix = getDistanceMatrix(new_exp, multipeptides, initial_alignment_cutoff)
        run_1 = [r for r in new_exp.runs if r.get_id() == rid][0]
        for run_0 in new_exp.runs:
            addDataToTrafo(tr_data, run_0, run_1, spl_aligner, multipeptides, options.realign_method, max_rt_diff)

    elif method == "singleShortestPath":
        dist_matrix = None

        tree = MinimumSpanningTree(getDistanceMatrix(new_exp, multipeptides, initial_alignment_cutoff))
        tree_mapped = [(new_exp.runs[a].get_id(), new_exp.runs[b].get_id()) for a,b in tree]
        for edge in tree:
            addDataToTrafo(tr_data, new_exp.runs[edge[0]], new_exp.runs[edge[1]], spl_aligner, multipeptides, options.realign_method, max_rt_diff)

    else:
        raise Exception("Unknown method: " + method)

    start = time.time()
    multipeptides = analyze_multipeptides(new_exp, multipeptides, swath_chromatograms,
        tr_data, options.border_option, rid, tree=tree_mapped, mat=dist_matrix)
    print("Analyzing the runs took %ss" % (time.time() - start) )

    return new_exp, multipeptides

def run_impute_values(options, peakgroups_file, trafo_fnames):
    """Impute values across chromatograms

    Args:
        peakgroups_file(filename): CSV file containing all peakgroups
        trafo_fnames(filename): A list of .tr filenames (it is assumed that in
            the same directory also the chromatogram mzML reside)
    Returns:
        A tuple of:
            new_exp(AlignmentExperiment): experiment containing the aligned peakgroups
            multipeptides(list(AlignmentHelper.Multipeptide)): list of multipeptides

    This function will read the csv file with all peakgroups as well as the
    transformation files (.tr) and the corresponding raw chromatograms which
    need to be in the same folder. It will then try to impute missing values
    for those peakgroups where no values is currently present, reading the raw
    chromatograms.

    It produces a dict swath_chromatograms which can be described as following:

        a dict whose key is the run_id (0_1, 0_2 etc) pointing to an all_swathes dict
            the all_swathes has a key which is the floor m/z of the *first* precursor found in that run and a value of type pymzml.run.Reader:
        { "0_0" : 
            {501 : pymzml.run.Reader,
             530 : pymzml.run.Reader,
                [...]
            }
          "0_1" : 
            {501 : pymzml.run.Reader,
             530 : pymzml.run.Reader,
                [...]
            }
            }
    """

    fdr_cutoff_all_pg = 1.0
    start = time.time()
    reader = SWATHScoringReader.newReader([peakgroups_file], options.file_format, readmethod="complete")
    new_exp = Experiment()
    new_exp.runs = reader.parse_files()
    multipeptides = new_exp.get_all_multipeptides(fdr_cutoff_all_pg, verbose=False)
    print("Parsing the peakgroups file took %ss" % (time.time() - start) )

    start = time.time()
    transformation_collection_ = transformations.TransformationCollection()
    for filename in trafo_fnames:
        transformation_collection_.readTransformationData(filename)

    # Read the datapoints and perform the smoothing
    print("Reading the trafo file took %ss" % (time.time() - start) )
    start = time.time()
    transformation_collection_.initialize_from_data(reverse=True, smoother=options.realign_method)
    print("Initializing the trafo file took %ss" % (time.time() - start) )

    if options.do_single_run and not options.dry_run:
        # Do only a single run : read only one single file
        start = time.time()
        swath_chromatograms = SwathChromatogramCollection()
        swath_chromatograms.parse([ options.do_single_run ])
        print("Reading the chromatogram files took %ss" % (time.time() - start) )
        assert len(swath_chromatograms.get_runids() ) == 1
        rid = swath_chromatograms.get_runids()[0]
        # 
        start = time.time()
        multipeptides = analyze_multipeptides(new_exp, multipeptides, swath_chromatograms,
            transformation_collection_, options.border_option, rid)
        print("Analyzing the runs took %ss" % (time.time() - start) )
        return new_exp, multipeptides

    swath_chromatograms = SwathChromatogramCollection()
    swath_chromatograms.parse(trafo_fnames)
    print("Reading the chromatogram files took %ss" % (time.time() - start) )

    if options.dry_run:
        print "Dry Run only"
        print "Found multipeptides:", len(multipeptides)
        print "Found swath chromatograms:", swath_chromatograms
        return [], []

    start = time.time()
    if options.cache_in_memory:
        run_ids = [r.get_id() for r in new_exp.runs]
        for rid in run_ids:
            # Create the cache for run "rid" and then only extract peakgroups from this run
            swath_chromatograms.createRunCache(rid)
            multipeptides = analyze_multipeptides(new_exp, multipeptides, 
                swath_chromatograms, transformation_collection_, options.border_option, rid)
    else:
        multipeptides = analyze_multipeptides(new_exp, multipeptides, swath_chromatograms,
            transformation_collection_, options.border_option)
    print("Analyzing the runs took %ss" % (time.time() - start) )
    return new_exp, multipeptides

def analyze_multipeptides(new_exp, multipeptides, swath_chromatograms, 
                          transformation_collection_, border_option, onlyExtractFromRun=None, tree=None, mat=None):
    """Analyze the multipeptides and impute missing values

    Args:
        new_exp(AlignmentExperiment): experiment containing the aligned peakgroups
        multipeptides(list(AlignmentHelper.Multipeptide)): list of multipeptides
        swath_chromatograms(dict): containing the objects pointing to the original chrom mzML (see run_impute_values)
        transformation_collection_(.TransformationCollection): specifying how to transform between retention times of different runs

    Returns:
        The updated multipeptides

    This function will update the input multipeptides and add peakgroups, imputing missing values 
    """

    # Go through all aligned peptides
    class CounterClass: pass
    cnt = CounterClass()
    cnt.integration_bnd_warnings = 0
    cnt.imputations = 0
    cnt.imputation_succ = 0
    cnt.peakgroups = 0

    print "Will work on %s peptides" % (len(multipeptides))
    for i,m in enumerate(multipeptides):
        if i % 500 == 0: print "Done with %s out of %s" % (i, len(multipeptides))

        # Ensure that each run has either exactly one peakgroup (or zero)
        if all ([pg.get_cluster_id() == -1 for p in m.get_peptides() for pg in p.get_all_peakgroups()]) and \
           any([len(p.peakgroups) > 1 for p in m.get_peptides()] ):
            raise Exception("Found more than one peakgroup for peptide %s - \
                \n is this after alignment or did you forget to run feature_alignment.py beforehand?" % (
                m.get_id() ))

        clusters = set( [pg.get_cluster_id() for p in m.get_peptides() for pg in p.get_all_peakgroups()])
        for cl in clusters:
            selected_pg = [pg for p in m.get_peptides() for pg in p.get_all_peakgroups() if pg.get_cluster_id() == cl]
            analyze_multipeptide_cluster(m, cnt, new_exp, swath_chromatograms, 
                                      transformation_collection_, border_option, selected_pg, cl,
                                      onlyExtractFromRun, tree, mat)

        if all ([pg.get_cluster_id() == -1 for p in m.get_peptides() for pg in p.get_all_peakgroups()]):
            # no cluster info was read in -> select all
            for p in m.get_peptides():
                for pg in p.get_all_peakgroups():
                    pg.select_this_peakgroup()

    print "Imputations:", cnt.imputations, "Successful:", cnt.imputation_succ, "Still missing", cnt.imputations - cnt.imputation_succ
    print "WARNING: %s times: Chromatogram does not cover full range"  % (cnt.integration_bnd_warnings)
    print "Peakgroups:", cnt.peakgroups
    return multipeptides 

def analyze_multipeptide_cluster(current_mpep, cnt, new_exp, swath_chromatograms, 
                          transformation_collection_, border_option, selected_pg, cluster_id,
                          onlyExtractFromRun=None, tree=None, mat=None):

        for rid in [r.get_id() for r in new_exp.runs]:
            cnt.peakgroups += 1

            # Check whether we have a peakgroup for this run id, if yes we mark
            # this peakgroup as selected for output, otherwise we try to impute.
            if not any([pg.peptide.run.get_id() == rid for pg in selected_pg]):
                cnt.imputations += 1
                imputed=True

                # Skip if we should not extract from this run
                if not onlyExtractFromRun is None:
                    if onlyExtractFromRun != rid:
                        continue

                # Select current run, compute right/left integration border and then integrate
                current_run = [r for r in new_exp.runs if r.get_id() == rid][0]
                rmap = dict([(r.get_id(),i) for i,r in enumerate(new_exp.runs) ])

                # print "Will try to fill NA in run", current_run.get_id(), "for peptide", current_mpep.get_peptides()[0].get_id()
                if tree is not None:
                    border_l, border_r = integrationBorderShortestPath(current_mpep, selected_pg, 
                        rid, transformation_collection_, tree, rmap)
                elif mat is not None:
                    border_l, border_r = integrationBorderShortestDistance(selected_pg, 
                        rid, transformation_collection_, mat, rmap)
                else:
                    border_l, border_r = integrationBorderReference(new_exp, selected_pg, rid, transformation_collection_, border_option)
                newpg = integrate_chromatogram(selected_pg[0], current_run, swath_chromatograms,
                                             border_l, border_r, cnt)
                if newpg != "NA": 
                    # print "Managed to fill NA in run", current_run.get_id(), \
                    #       "with value", newpg.get_value("Intensity"), "/ borders", border_l, border_r
                    cnt.imputation_succ += 1
                    precursor = GeneralPrecursor(newpg.get_value("transition_group_id"), current_run)
                    # Select for output, add to precursor
                    newpg.setClusterID(cluster_id)
                    precursor.add_peakgroup(newpg)
                    current_mpep.insert(rid, precursor)

def integrationBorderShortestPath(m, selected_pg, target_run, transformation_collection_, tree, rmap):
    """Determine the optimal integration border by using the shortest path in the MST

    Returns:
        A tuple of (left_integration_border, right_integration_border)
    """

    # Get the shortest path through the MST
    available_runs = [pg.peptide.run.get_id() for pg in selected_pg if pg.get_fdr_score() < 1.0]
    final_path = graphs.findShortestMSTPath(tree, target_run, available_runs)

    # Extract the starting point and the starting borders (left/right)
    source_run = final_path[-1]
    best_pg = [pg for pg in selected_pg if pg.peptide.run.get_id() == source_run][0]
    rwidth = float(best_pg.get_value("rightWidth"))
    lwidth = float(best_pg.get_value("leftWidth"))
    # if verb: print "Compare from", source_run, "directly to", target_run,\
    #     transformation_collection_.getTrafo(source_run, target_run).predict([lwidth])[0], \
    #     transformation_collection_.getTrafo(source_run, target_run).predict([rwidth])[0]

    # Traverse path in reverse order
    for target_run_ in reversed(final_path[:-1]):

        # Transform, go to next step
        lwidth = transformation_collection_.getTrafo(source_run, target_run_).predict([lwidth])[0]
        rwidth = transformation_collection_.getTrafo(source_run, target_run_).predict([rwidth])[0]
        source_run = target_run_

    return lwidth, rwidth

def integrationBorderShortestDistance(selected_pg, target_run, transformation_collection_, mat, rmap):
    """Determine the optimal integration border by using the shortest distance (direct transformation)

    Returns:
        A tuple of (left_integration_border, right_integration_border)
    """

    # Available runs which have a reliable RT value (no noise integration values ... )
    available_runs = [pg.peptide.run.get_id() for pg in selected_pg if pg.get_fdr_score() < 1.0]

    # Select current row from the matrix, reduce to columns of runs for which
    # we actually have a value and select closest among these
    current_matrix_row = mat[rmap[target_run],]
    current_matrix_row = [ (current_matrix_row[ rmap[curr] ], curr) for curr in available_runs]
    source_run = min(current_matrix_row)[1]

    # Transform
    best_pg = [pg for pg in selected_pg if pg.peptide.run.get_id() == source_run][0]
    rwidth = float(best_pg.get_value("rightWidth"))
    lwidth = float(best_pg.get_value("leftWidth"))
    leftW = transformation_collection_.getTrafo(source_run, target_run).predict([lwidth])[0]
    rightW = transformation_collection_.getTrafo(source_run, target_run).predict([rwidth])[0]
    return leftW, rightW

def integrationBorderReference(new_exp, selected_pg, rid, transformation_collection_, border_option):
    """Determine the optimal integration border by taking the mean of all other peakgroup boundaries using a reference run.

    Args:
        new_exp(AlignmentExperiment): experiment containing the aligned peakgroups
        selected_pg(list(GeneralPeakGroup)): list of selected peakgroups (e.g. those passing the quality threshold)
        rid(String): current run id
        transformation_collection_(.TransformationCollection): specifying how to transform between retention times of different runs

    Returns:
        A tuple of (left_integration_border, right_integration_border) in the retention time space of the _reference_ run
    """
    current_run = [r for r in new_exp.runs if r.get_id() == rid][0]
    ref_id = transformation_collection_.getReferenceRunID()

    pg_lefts = []
    pg_rights = []
    for pg in selected_pg:

        rwidth = float(pg.get_value("rightWidth"))
        lwidth = float(pg.get_value("leftWidth"))
        this_run_rwidth = ImputeValuesHelper.convert_to_this(pg.peptide.run.get_id(),
            current_run.get_id(), ref_id, rwidth, transformation_collection_)
        this_run_lwidth = ImputeValuesHelper.convert_to_this(pg.peptide.run.get_id(),
            current_run.get_id(), ref_id, lwidth, transformation_collection_)

        pg_lefts.append(this_run_lwidth)
        pg_rights.append(this_run_rwidth)

    if border_option == "mean":
        integration_left = numpy.mean(pg_lefts)
        integration_right = numpy.mean(pg_rights)
    elif border_option == "median":
        integration_left = numpy.median(pg_lefts)
        integration_right = numpy.median(pg_rights)
    elif border_option == "max_width":
        integration_left = numpy.min(pg_lefts)
        integration_right = numpy.max(pg_rights)
    else:
        raise Exception("Unknown border determination option %s" % border_option)

    std_warning_level = 20
    if numpy.std(pg_rights) > std_warning_level or numpy.std(pg_lefts) > std_warning_level: 
        pass
        # TODO : what if they are not consistent ?
        # print "Std is too large", pg_rights, pg_lefts
        # print "overall mean, std", numpy.mean(pg_lefts), numpy.std(pg_lefts), numpy.mean(pg_rights), numpy.std(pg_rights)

    return integration_left, integration_right

def integrate_chromatogram(template_pg, current_run, swath_chromatograms, 
                           left_start, right_end, cnt):
    """ Integrate a chromatogram from left_start to right_end and store the sum.

    Args:
        template_pg(GeneralPeakGroup): A template peakgroup from which to construct the new peakgroup
        current_run(SWATHScoringReader.Run): current run where the missing value occured
        swath_chromatograms(dict): containing the objects pointing to the original chrom mzML
        left_start(float): retention time for integration (left border)
        right_end(float): retention time for integration (right border)

    Returns:
        A new GeneralPeakGroup which contains the new, integrated intensity for this run (or "NA" if no chromatograms could be found).

    Create a new peakgroup from the old pg and then store the integrated intensity.
    """

    current_rid = current_run.get_id()
    orig_filename = current_run.get_openswath_filename()
    aligned_filename = current_run.get_aligned_filename()

    # Create new peakgroup by copying the old one
    newrow = ["NA" for ele in template_pg.row]
    newpg = GeneralPeakGroup(newrow, current_run, template_pg.peptide)
    for element in ["transition_group_id"]:
        newpg.set_value(element, template_pg.get_value(element))
    for element in ["decoy", "Sequence", "FullPeptideName", "Charge", "ProteinName", "nr_peaks", "m.z", "m/z"]:
        try:
            newpg.set_value(element, template_pg.get_value(element))
        except KeyError:
            # These are non-essential column names, we can just skip them ...
            pass

    newpg.set_value("align_runid", current_rid)
    newpg.set_value("run_id", current_rid)
    newpg.set_value("filename", orig_filename)
    newpg.set_value("align_origfilename", aligned_filename)
    newpg.set_value("RT", (left_start + right_end) / 2.0 )
    newpg.set_value("leftWidth", left_start)
    newpg.set_value("rightWidth", right_end)
    newpg.set_value("m_score", 2.0)
    newpg.set_value("d_score", '-10') # -10 has a p value of 1.0 for 1-right side cdf
    newpg.set_value("peak_group_rank", -1)

    import uuid 
    thisid = str(uuid.uuid1() )
    newpg.set_normalized_retentiontime((left_start + right_end) / 2.0 )
    newpg.set_fdr_score(1.0)
    newpg.set_feature_id(thisid)

    integrated_sum = 0
    chrom_ids = template_pg.get_value("aggr_Fragment_Annotation").split(";")
    peak_areas = []
    for chrom_id in chrom_ids:
        chromatogram = swath_chromatograms.getChromatogram(current_rid, chrom_id)
        # Check if we got a chromatogram with at least 1 peak: 
        if chromatogram is None:
            print "chromatogram is None (tried to get %s from run %s)" % (chrom_id, current_rid)
            return "NA"
        if len(chromatogram.peaks) == 0:
            print "chromatogram has no peaks None (tried to get %s from run %s)" % (chrom_id, current_rid)
            return "NA"

        # Check whether we even have data between left_start and right_end ... 
        if chromatogram.peaks[0][0] > left_start or chromatogram.peaks[-1][0] < right_end:
            cnt.integration_bnd_warnings += 1
            # print "WARNING: Chromatogram (%s,%s) does not cover full range (%s,%s)"  % (
            #     chromatogram.peaks[0][0],chromatogram.peaks[-1][0], left_start, right_end)

        curr_int = sum( [p[1] for p in chromatogram.peaks if p[0] > left_start and p[0] < right_end ])
        peak_areas.append(curr_int)
        # print integrated_sum, "chromatogram", sum(p[1] for p in chromatogram.peaks), \
        # "integrated from \t%s\t%s in run %s" %( left_start, right_end, current_rid), newpg.get_value("transition_group_id")

    integrated_sum = sum(peak_areas)
    aggr_annotation = ";".join(chrom_ids)
    aggr_peak_areas = ";".join([str(it) for it in peak_areas])
    newpg.set_value("aggr_Peak_Area", aggr_peak_areas)
    newpg.set_value("aggr_Fragment_Annotation", aggr_annotation)
    newpg.set_value("Intensity", integrated_sum)
    newpg.set_intensity(integrated_sum)
    return newpg

def write_out(new_exp, multipeptides, outfile, matrix_outfile, single_outfile):
    """ Write the result to disk 
    """
    # write out the complete original files 
    writer = csv.writer(open(outfile, "w"), delimiter="\t")
    header_first = new_exp.runs[0].header
    for run in new_exp.runs:
        assert header_first == run.header
    writer.writerow(header_first)

    print "number of precursors quantified:", len(multipeptides)
    for m in multipeptides:
        # selected_peakgroups = [p.peakgroups[0] for p in m.get_peptides()]
        # if (len(selected_peakgroups)*2.0 / len(new_exp.runs) < fraction_needed_selected) : continue
        for p in m.get_peptides():
            selected_pg = p.peakgroups[0]
            if selected_pg is None: 
                continue
            if single_outfile:
                # Only write the newly imputed ones ... 
                if float(selected_pg.get_value("m_score")) > 1.0:
                    row_to_write = selected_pg.row
                    writer.writerow(row_to_write)
            else:
                row_to_write = selected_pg.row
                writer.writerow(row_to_write)

    if len(matrix_outfile) > 0:
        write_out_matrix_file(matrix_outfile, new_exp.runs, multipeptides, 0.0, style=options.matrix_output_method)

def handle_args():
    usage = "" #usage: %prog --in \"files1 file2 file3 ...\" [options]" 
    usage += "\nThis program will impute missing values"

    parser = argparse.ArgumentParser(description = usage )
    parser.add_argument('--in', dest="infiles", nargs = '+', required=False, help = 'A list of transformation files in the same folder as the .chrom.mzML files')
    parser.add_argument("--peakgroups_infile", dest="peakgroups_infile", required=True, help="Infile containing peakgroups (outfile from feature_alignment.py)")
    parser.add_argument("--out", dest="output", required=True, help="Output file with imputed values")
    parser.add_argument('--file_format', default='openswath', help="Which input file format is used (openswath or peakview)")
    parser.add_argument("--out_matrix", dest="matrix_outfile", default="", help="Matrix containing one peak group per row (supports .csv, .tsv or .xls)")
    parser.add_argument("--matrix_output_method", dest="matrix_output_method", default='none', help="Which columns are written besides Intensity (none, RT, score, source or full)")
    parser.add_argument('--border_option', default='median', metavar="median", help="How to determine integration border (possible values: max_width, mean, median). Max width will use the maximal possible width (most conservative since it will overestimate the background signal).")
    parser.add_argument('--dry_run', action='store_true', default=False, help="Perform a dry run only")
    parser.add_argument('--cache_in_memory', action='store_true', default=False, help="Cache data from a single run in memory")
    parser.add_argument('--method', dest='method', default="allTrafo", help="Which method to use (singleShortestPath, singleClosestRun, other)")
    parser.add_argument('--realign_runs', dest='realign_method', default="splineR", help="How to re-align runs in retention time ('diRT': use only deltaiRT from the input file, 'linear': perform a linear regression using best peakgroups, 'splineR': perform a spline fit using R, 'splineR_external': perform a spline fit using R (start an R process using the command line, 'splinePy' use Python native spline from scikits.datasmooth (slow!), 'lowess': use Robust locally weighted regression (lowess smoother)")
    parser.add_argument('--verbosity', default=0, type=int, help="Verbosity")
    parser.add_argument('--do_single_run', default='', metavar="", help="Only do a single run")

    experimental_parser = parser.add_argument_group('experimental options')

    args = parser.parse_args(sys.argv[1:])
    args.verbosity = int(args.verbosity)
    return args

def main(options):
    import time
    if options.method in ["singleShortestPath", "singleClosestRun"]:
        new_exp, multipeptides = runSingleFileImputation(options, options.peakgroups_infile, options.do_single_run, options.method)
    else:
        new_exp, multipeptides = run_impute_values(options, options.peakgroups_infile, options.infiles)
    if options.dry_run: return
    write_out(new_exp, multipeptides, options.output, options.matrix_outfile, options.do_single_run)

if __name__=="__main__":
    options = handle_args()
    main(options)

