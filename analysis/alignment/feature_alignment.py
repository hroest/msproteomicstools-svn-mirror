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
"""

import sys, csv
import numpy
from msproteomicstoolslib.math.chauvenet import chauvenet
from msproteomicstoolslib.format.SWATHScoringReader import *
from sys import stdout

verb = True
verb = False

#infiles = ['split_napedro_L120227_009_SW_mprophet_all_peakgroups.xls', 'split_napedro_L120227_010_SW_mprophet_all_peakgroups.xls']
#infiles = ['split_napedro_L120420_006_SW_mprophet_all_peakgroups.xls', 'split_napedro_L120420_007_SW_mprophet_all_peakgroups.xls', 'split_napedro_L120420_008_SW_mprophet_all_peakgroups.xls', 'split_napedro_L120420_009_SW_mprophet_all_peakgroups.xls', 'split_napedro_L120420_010_SW_mprophet_all_peakgroups.xls']



"""
Testing

time python feature_alignment.py --method best_cluster_score --in /tmp/testdata.csv --out /tmp/align_out --fdr_cutoff 0.01  --max_rt_diff 20 --max_fdr_quality 0.35  

for data, see hroest usernas: ~/html/code/testdata/feature_alignment

$ time python feature_alignment.py --method best_overall --in testdata_align/big/strep0_repl1_r02.minimal.xls testdata_align/big/strep0_repl1_r03.minimal.xls --out /tmp/align_out --fdr_cutoff 0.01  --max_rt_diff 30 --max_fdr_quality 0.2  --outlier_thresh 30 && diff /tmp/align_out_idsonly.csv testdata_align/big/expected_ids.csv  | wc
Parsing input files
Reading testdata_align/big/strep0_repl1_r03.minimal.xls, processing run 0_1
Found 2 runs
Parsing run 2 out of 2
===================================
Finished parsing, number of precursors and peptides per run
All precursors [10576, 9895] (union of all runs 11634)
All proteins [1302, 1244] (union of all runs 1424)
Present in all runs 8837
===========================================================================
===========================================================================
Total we have 2 runs with 11634 peakgroups quantified in at least one run, giving maximally nr peakgroups 23268
We were able to quantify 10429 / 11634 precursors in all runs (up from 8837 before alignment)
We were able to quantify 7914 peptides and 1310 proteins in all runs (up from 1077 before alignment)
Able to quantify 22063 / 23268 of which we aligned 1592 and changed order of 275 and could not align 1205

real    0m19.942s
user    0m19.117s
sys     0m0.684s
      0       0       0

$ time python feature_alignment.py --method best_cluster_score --in testdata_align/big/strep0_repl1_r02.minimal.xls testdata_align/big/strep0_repl1_r03.minimal.xls --out /tmp/align_out --fdr_cutoff 0.01  --max_rt_diff 30 --max_fdr_quality 0.2  --outlier_thresh 30 && diff /tmp/align_out_idsonly.csv testdata_align/big/cluster_expected_ids.csv  | wc
Parsing input files
Reading file testdata_align/big/strep0_repl1_r03.minimal.xls
Found 2 runs
Parsing run 2 out of 2
===================================
Finished parsing, number of precursors and peptides per run
All precursors [10576, 9895] (union of all runs 11634)
All proteins [1302, 1244] (union of all runs 1424)
Present in all runs 8837
===========================================================================
===========================================================================
Total we have 2 runs with 11634 peakgroups quantified in at least one run, giving maximally nr peakgroups 23268
We were able to quantify 10748 / 11634 precursors in all runs (up from 8837 before alignment)
We were able to quantify 8149 peptides and 1347 proteins in all runs (up from 1077 before alignment)
Able to quantify 22382 / 23268 of which we aligned 1944 and changed order of 261 and could not align 0

real    0m17.469s
user    0m17.013s
sys     0m0.340s
      0       0       0


infiles = [
'strep_align/Strep0_Repl1_R02/minimal.xls',
'strep_align/Strep0_Repl1_R03/minimal.xls',
'strep_align/Strep0_Repl2_R02/minimal.xls',
'strep_align/Strep0_Repl2_R03/minimal.xls'
] 

infiles = [
'strep_align/Strep0_Repl1_R02/split_hroest_K120808_all_peakgroups_aligned.xls',
'strep_align/Strep0_Repl1_R03/split_hroest_K120808_all_peakgroups_aligned.xls'
]

from feature_alignment import *
infiles = [
'strep_align/Strep0_Repl1_R02/split_hroest_K120808_all_peakgroups_aligned.xls',
'strep_align/Strep0_Repl1_R03/split_hroest_K120808_all_peakgroups_aligned.xls',
'strep_align/Strep0_Repl2_R02/split_hroest_K120808_all_peakgroups_aligned.xls',
'strep_align/Strep0_Repl2_R03/split_hroest_K120808_all_peakgroups_aligned.xls'
] 


this_exp = Experiment()
this_exp.parse_files(infiles, "openswath", True)

# Map the precursors across multiple runs, determine the number of
# precursors in all runs without alignment.
multipeptides = this_exp.get_all_multipeptides(0.01)

this_exp.rt_align_all_runs(multipeptides, alignment_fdr_threshold = 0.0001, use_scikit=False)

alignment = this_exp.align_features(multipeptides, 30, 0.01, 0.2)

"""

class Multipeptide():
    """
    A collection of the same precursors (chromatograms) across multiple runs
    """
  
    def __init__(self):
        self.peptides = {}
        self._has_null = False

    def __str__(self):
        return "Precursors of % runs." % len(self.peptides)
  
    def insert(self, runid, peptide):
      if peptide is None: self._has_null = True; return
      self.peptides[runid] = peptide
    
    def all_above_cutoff(self, cutoff):
      for p in self.peptides.values():
        if p.get_best_peakgroup().get_fdr_score() > cutoff: 
            return False
      return True
  
    def get_id(self):
      return self.peptides.values()[0].get_id()
  
    def all_below_cutoff(self, cutoff):
      for p in self.peptides.values():
        if p.get_best_peakgroup().get_fdr_score() < cutoff: return False
      return True
  
    def all_selected(self):
      for p in self.peptides.values():
          if p.get_selected_peakgroup() is None: return False
      return True

    def get_decoy(self):
        # If one is a decoy, all of them are...
        if len(self.peptides.values()) > 1:
            return self.peptides.values()[0].get_decoy() 
        return False

    def get_selected_peakgroups(self):
      return [p.get_selected_peakgroup() for p in self.peptides.values() if p.get_selected_peakgroup() is not None]

    def find_best_peptide_pg(self):
      best_fdr = 1.0
      for p in self.peptides.values():
        if(p.get_best_peakgroup().get_fdr_score() < best_fdr): 
            result = p.get_best_peakgroup()
            best_fdr = p.get_best_peakgroup().get_fdr_score() 
      return result

    def has_null_peptides(self):
      return self._has_null
  
    def detect_outliers(self):
        # Uses chauvenet's criterion for outlier detection to find peptides
        # whose retention time is different from the rest.
        rts = [float(p.get_selected_peakgroup().get_normalized_retentiontime()) for p in self.peptides.values() if p.get_selected_peakgroup() is not None]
        runids = numpy.array([p.get_selected_peakgroup().get_run_id() for p in self.peptides.values() if p.get_selected_peakgroup() is not None])
        if len(rts) == 1: return []
        outliers = chauvenet(numpy.array(rts),numpy.array(rts))
        return runids[~outliers]

class SplineAligner():
    """
    Use the datasmoothing part of msproteomicstoolslib to align 2 runs in
    retention times using splines.
    """
    def __init__(self):
      pass

    def determine_best_run(self, experiment, alignment_fdr_threshold):

        maxcount = -1
        bestrun = -1
        for run in experiment.runs:
            cnt = 0
            for peptide in run.all_peptides.values():
                if peptide.get_decoy(): continue
                pg = peptide.get_best_peakgroup()
                if pg.get_fdr_score() < alignment_fdr_threshold:
                    cnt += 1
            if cnt > maxcount:
                maxcount = cnt
                bestrun = run.get_id()
        print "found best run", bestrun, "with features", maxcount
        return [r for r in experiment.runs if r.get_id() == bestrun][0]

    def spline_align_runs(self, bestrun, run, multipeptides, alignment_fdr_threshold, use_scikit):
        import msproteomicstoolslib.math.Smoothing as smoothing
        sm = smoothing.Smoothing()

        # get those peptides we want to use for alignment => for this use the mapping
        data1 = []
        data2 = []
        for m in multipeptides:
            ref_pep = m.peptides[bestrun.get_id()].get_best_peakgroup()
            align_pep = m.peptides[run.get_id()].get_best_peakgroup()
            if ref_pep.peptide.get_decoy() or align_pep.peptide.get_decoy(): continue
            if ref_pep.get_fdr_score() < alignment_fdr_threshold and align_pep.get_fdr_score() < alignment_fdr_threshold:
                data1.append(ref_pep.get_normalized_retentiontime())
                data2.append(align_pep.get_normalized_retentiontime())

        print "Will align run %s against %s, using %s features" % (run.get_id(), bestrun.get_id(), len(data1))

        all_pg = []
        for pep in run.all_peptides.values():
            all_pg.extend( [ (pg.get_normalized_retentiontime(), pg.get_feature_id()) for pg in pep.get_all_peakgroups()] )

        rt_eval = [ pg[0] for pg in all_pg]

        # data1 is master, data2 is slave. Since we want to predict how to
        # convert from slave to master, slave is first and master is second
        try:
            if use_scikit: import dummydummy # forces to use scikit
            aligned_result = sm.smooth_spline_r(data2, data1, rt_eval)
        except ImportError:
          aligned_result = sm.smooth_spline_scikit_wrap(data2, data1, rt_eval)
          print "use scikit to compute spline alignment..."

        # The two methods produce very, very similar results
        # but R is faster => prefer to use R when possible.
        # hist(aligned_result - aligned_result_2, 100)
        # numpy.std(aligned_result - aligned_result_2)
        # 0.66102016517870454
        # numpy.median(aligned_result - aligned_result_2)
        # -0.020456989235640322

        # now re-populate the peptide data!
        i = 0
        for pep in run.all_peptides.values():
            mutable = [list(pg) for pg in pep.peakgroups_]
            for k in range(len(mutable)):
                mutable[k][2] = aligned_result[i]
                i += 1
            pep.peakgroups_ = [ tuple(m) for m in mutable]

class Experiment():
    """
    An Experiment is a container for multiple experimental runs - some of which may contain the same precursors.
    """

    def __init__(self):
        self.runs = []

    def rt_align_all_runs(self, multipeptides, alignment_fdr_threshold = 0.0001, use_scikit=False):

        print "Will re-align runs"
        spl_aligner = SplineAligner()

        # get the best run (e.g. the one with the most ids below threshold)
        bestrun = spl_aligner.determine_best_run(self, alignment_fdr_threshold)

        # go through all runs and align two runs at a time
        for run in self.runs:
            if run.get_id() == bestrun.get_id(): continue # do not align reference run itself
            spl_aligner.spline_align_runs(bestrun, run, multipeptides, alignment_fdr_threshold, use_scikit)

    def get_all_multipeptides(self, fdr_cutoff):
        # Find all precursors that are above the fdr cutoff in each run and
        # build a union of those precursors. Then search for each of those
        # precursors in all the other runs and build a multipeptide /
        # multiprecursor.
        union_transition_groups = []
        union_proteins = []
        for i,r in enumerate(self.runs):
            stdout.write("\rParsing run %s out of %s" % (i+1, len(self.runs) ))
            stdout.flush()
            union_transition_groups.append( [peak.peptide.get_id() for peak in r.get_best_peaks_with_cutoff(fdr_cutoff) if not peak.peptide.get_decoy()] )
            union_proteins.append( list(set([peak.peptide.protein_name for peak in r.get_best_peaks_with_cutoff(fdr_cutoff) if not peak.peptide.get_decoy()])) )
        stdout.write("\r\r\n") # clean up

        self.union_transition_groups_set = set(union_transition_groups[0])
        self.union_proteins_set = set(union_proteins[0])
        for groups in union_transition_groups:
          self.union_transition_groups_set = self.union_transition_groups_set.union( groups )
        for proteins in union_proteins:
          self.union_proteins_set = self.union_proteins_set.union( proteins )

        print "==================================="
        print "Finished parsing, number of precursors and peptides per run"
        print "All target precursors", [len(s) for s in union_transition_groups], "(union of all runs %s)" % len(self.union_transition_groups_set)
        print "All target proteins", [len(s) for s in union_proteins], "(union of all runs %s)" % len(self.union_proteins_set)

        multipeptides = []
        for peptide_id in self.union_transition_groups_set:
          m = Multipeptide()
          for r in self.runs:
            m.insert(r.get_id(), r.get_peptide(peptide_id))
          multipeptides.append(m)
        return multipeptides

    def get_max_pg(self):
      return len(self.runs)*len(self.union_transition_groups_set)

    def print_stats(self, multipeptides, alignment, outlier_detection, fdr_cutoff):
        # Do statistics and print out
        in_all_runs_wo_align = len([1 for m in multipeptides if m.all_above_cutoff(fdr_cutoff)])
        proteins_in_all_runs_wo_align = len(set([m.find_best_peptide_pg().peptide.protein_name for m in multipeptides if m.all_above_cutoff(fdr_cutoff)]))
        proteins_in_all_runs_wo_align_target = len(set([m.find_best_peptide_pg().peptide.protein_name for m in multipeptides if m.all_above_cutoff(fdr_cutoff) and not m.find_best_peptide_pg().peptide.get_decoy()]))

        print "Present in all runs", in_all_runs_wo_align
        precursors_in_all_runs = [m for m in multipeptides if m.all_selected()]
        nr_peptides = len(set([prec.find_best_peptide_pg().peptide.sequence for prec in precursors_in_all_runs]))
        nr_proteins = len(set([prec.find_best_peptide_pg().peptide.protein_name for prec in precursors_in_all_runs]))
        nr_peptides_target = len(set([prec.find_best_peptide_pg().peptide.sequence for prec in precursors_in_all_runs if not prec.find_best_peptide_pg().peptide.get_decoy()]))
        nr_proteins_target = len(set([prec.find_best_peptide_pg().peptide.protein_name for prec in precursors_in_all_runs if not prec.find_best_peptide_pg().peptide.get_decoy()]))
        nr_precursors_in_all = len([1 for m in multipeptides if m.all_selected() and not m.get_decoy()])
        max_pg = self.get_max_pg()
        print "="*75
        print "="*75
        print "Total we have", len(self.runs), "runs with", len(self.union_transition_groups_set), "peakgroups quantified in at least one run, " + \
                "giving maximally nr peakgroups", max_pg
        print "We were able to quantify", nr_precursors_in_all, "/", len(self.union_transition_groups_set), "precursors in all runs (up from", in_all_runs_wo_align, "before alignment)"
        #print "We were able to quantify", nr_peptides, "peptides and", nr_proteins, "proteins in all runs (up from", proteins_in_all_runs_wo_align, "before alignment)"
        print "We were able to quantify", nr_peptides_target, "target peptides and", nr_proteins_target, "target proteins in all runs (up from target", proteins_in_all_runs_wo_align_target, "before alignment)"
        print "Able to quantify", alignment.nr_quantified, "/", max_pg, "of which we aligned", alignment.nr_aligned, "and changed order of", alignment.nr_changed, "and could not align", alignment.could_not_align
        if outlier_detection is not None: 
            print "Outliers:", outlier_detection.nr_outliers, "outliers in", len(multipeptides), "peptides or", outlier_detection.outlier_pg, "peakgroups out of", alignment.nr_quantified, "changed", outlier_detection.outliers_changed

    def write_to_file(self, multipeptides, infiles, outfile, fraction_needed_selected, file_format):
        writer = csv.writer(open(outfile + "_idsonly.csv", "w"), delimiter="\t")
        selected_ids = []
        for m in multipeptides:
            selected_peakgroups = m.get_selected_peakgroups()
            if (len(selected_peakgroups)*1.0 / len(self.runs) < fraction_needed_selected) : continue
            for p in m.peptides.values():
                selected_pg = p.get_selected_peakgroup()
                if selected_pg is None: continue
                writer.writerow([selected_pg.get_feature_id()])
                selected_ids.append(selected_pg.get_feature_id())
        selected_ids_set = set(selected_ids)

        # write out ids
        writer = csv.writer(open(outfile, "w"), delimiter="\t")
        header_first = self.runs[0].header
        for run in self.runs:
            assert header_first == run.header
        header_first += ["align_runid", "align_origfilename"]
        writer.writerow(header_first)

        writer = csv.writer(open(outfile + "_matrix.csv", "w"), delimiter="\t")
        run_ids = [r.get_id() for r in self.runs]
        header = ["Peptide"]
        for rid in run_ids:
            header.extend(["Intensity_%s" % rid, "RT_%s" % rid])
        writer.writerow(header)
        for m in multipeptides:
            line = [m.get_id()]
            for rid in run_ids:
                pg = m.peptides[rid].get_selected_peakgroup()
                if pg is None:
                    line.extend(["NA", "NA"])
                else:
                    line.extend([pg.get_intensity(), pg.get_normalized_retentiontime()])
            writer.writerow(line)

        # only in openswath we have the ID and can go back to the original file ... 
        if file_format != "openswath": return

        for file_nr, f in enumerate(infiles):
          header_dict = {}
          reader = csv.reader(open(f), delimiter="\t")
          header = reader.next()
          for i,n in enumerate(header):
            header_dict[n] = i
          for row in reader:
              if row[ header_dict["id"]] in selected_ids_set:
                  row_to_write = row
                  row_to_write += [file_nr, f]
                  writer.writerow(row_to_write)

  
class Cluster:
    """
    A representation of a cluster (used in align_features)
    """

    def __init__(self, peakgroups):
        self.peakgroups = peakgroups

    def select_one_per_run(self):
      # make sure for each cluster that we only have one peakgroup from each run --> take the best one
      run_ids = {}
      if verb: print "len pg ", len(self.peakgroups)
      for pg in self.peakgroups:
          rid = pg.peptide.get_run_id()
          if rid in run_ids:
              if run_ids[rid].get_fdr_score() > pg.get_fdr_score():
                  run_ids[rid] = pg
          else: run_ids[rid] = pg
      self.peakgroups = run_ids.values()

    def get_total_score(self):
      mult = 1
      for pg in self.peakgroups:
        mult = mult * pg.get_fdr_score()
      return mult

# Align features goes through all multipeptides (which contains peptides from all runs) and 
# tries to re-align them
def align_features(multipeptides, rt_diff_cutoff, fdr_cutoff, aligned_fdr_cutoff, method="best_overall"):
    class Alignment:
        def __init__(self): pass
    a = Alignment()
    br = False
    a.nr_aligned = 0
    a.nr_changed = 0
    a.nr_quantified = 0
    a.could_not_align = 0
    i = 0

    for m in multipeptides:
        # If the peptide is found in each run above the FDR, we are fine.
        # else we will try to realign some of the peakgroups.
        if verb: print "00000000000000000000000000000000000 new peptide ", m.peptides.values()[0].sequence
        if m.all_above_cutoff(fdr_cutoff):
          if verb: print "all above cutoff"
          for p in m.peptides.values():
              p.get_best_peakgroup().select_this_peakgroup()
              a.nr_quantified += 1
          continue
        elif method == "best_cluster_score":
            # i) get all RTs above the cutoff

          for p in m.peptides.values(): # loop over runs
              pg = p.get_best_peakgroup()
              if verb: print "best rt", pg.get_normalized_retentiontime(), pg.peptide.run.get_id(), pg.get_fdr_score()
          
          groups = []
          for p in m.peptides.values(): # loop over runs
              # use all peakgroups for clustering
              all_pgs = [ MinimalPeakGroup(result[0], result[1], result[2], p.selected_[index], p) for index, result in enumerate(p.peakgroups_)]
              for pg in all_pgs:
                  if pg.get_fdr_score() < aligned_fdr_cutoff: 
                      groups.append(pg) 
                      # if verb: print "group", pg.get_normalized_retentiontime(), pg.peptide.run.get_id(), pg.get_fdr_score()

          # do the clustering
          from cluster import HierarchicalClustering
          cl = HierarchicalClustering(groups, lambda x,y: abs(x.get_normalized_retentiontime()-y.get_normalized_retentiontime()))
          clusters_rt = cl.getlevel(rt_diff_cutoff)
          clusters_rt_obj = [Cluster(c) for c in clusters_rt]
          # if there was only one group, we need to prepare a special object of size one
          if len(groups) == 1: clusters_rt_obj = [Cluster( groups )]

          if verb: print "==== Clusters "
          # make sure only one is selected from each run...
          for c in clusters_rt_obj: 
              c.select_one_per_run()
              if verb:
                  print " - Cluster", c.get_total_score()
                  for pg in c.peakgroups: print pg.get_normalized_retentiontime(), pg.peptide.run.get_id()
            
          if len(clusters_rt_obj) == 1 :
              # great, only one cluster
              bestcluster = clusters_rt_obj[0]
          else:
              # select the best cluster
              bestcluster = clusters_rt_obj[0] # self.getBestCluster(clusters_rt, rt_maximal_distance, method = clusterMode)
              for cluster in clusters_rt_obj:
                  if cluster.get_total_score() < bestcluster.get_total_score(): bestcluster = cluster

          for pg in bestcluster.peakgroups:
            a.nr_quantified += 1
            pg.select_this_peakgroup()
            if pg.get_fdr_score() > fdr_cutoff:
                a.nr_aligned += 1
                if pg.get_normalized_retentiontime() != pg.peptide.get_best_peakgroup().get_normalized_retentiontime():
                    a.nr_changed += 1
                    if verb: print "FDR new align", pg.peptide.get_best_peakgroup().print_out(), "\tnew ====> ", pg.print_out()
                else:
                    if verb: print "FDR boost", pg.peptide.get_best_peakgroup().print_out(), " old ====> ", pg.print_out()
            else:
              if verb: print "no need to align"

          i += 1
        elif method == "best_overall":
          # If we just choose the cluster with the "best" peptide, we find find the best peptide over all runs
          best = m.find_best_peptide_pg()
          best_rt_diff = best.get_normalized_retentiontime()
          if verb: print "=====\nFDR best", best.print_out() #best.run.get_id(), "/", best.get_id(), best.get_fdr_score(), best.get_normalized_retentiontime(), best.get_value("RT"), best.get_value("rt_score") # rt_score = delta iRT
          for p in m.peptides.values(): # loop over runs
              pg = p.get_best_peakgroup()
              if pg.get_fdr_score() < fdr_cutoff:
                  pg.select_this_peakgroup()
                  a.nr_quantified += 1
                  if verb: print "FDR below", p.get_id(), pg.get_fdr_score(), pg.get_normalized_retentiontime() #, pg.get_value("RT"), pg.get_value("rt_score") # rt_score = delta iRT
              else:
                  # In this run, the peptide is above the FDR cutoff.
                  # i)  determine for this peptide p the "best" peakgroup based on the
                  #     alignment (find_closest_in_iRT)
                  # ii) use some minimal criteria for the feature to be
                  #     aligned, e.g. maximal rt_difference and maximal fdr / score of
                  #     the feature to be aligned.
                  newpg = p.find_closest_in_iRT(best_rt_diff)
                  if(
                      # use if the distance in RT is below the rt_diff_cutoff AND the fdr is below the aligned_fdr_cutoff
                      abs(float(newpg.get_normalized_retentiontime()) - float(best_rt_diff)) < rt_diff_cutoff and 
                       newpg.get_fdr_score() < aligned_fdr_cutoff):
                      a.nr_aligned += 1
                      a.nr_quantified += 1
                      newpg.select_this_peakgroup()
                      if pg != newpg:
                        a.nr_changed += 1
                        if verb: print "FDR new align", pg.print_out(), "\tnew ====> ", newpg.print_out()
                      else:
                        if verb: print "FDR boost", pg.print_out(), " old ====> ", newpg.print_out()
                  else:
                      if verb: print "could not align"
                      a.could_not_align += 1
        else:
            raise Exception("Method '%s' unknown" % method)
    return a

# Detect outliers in "good" groups
def detect_outliers(self, multipeptides, aligned_fdr_cutoff, outlier_threshold_seconds):
    class Outliers:
        def __init__(self): pass
    o = Outliers
    o.outliers_changed = 0
    o.outlier_pg = 0
    outliers = 0
    for m in multipeptides:
      out = m.detect_outliers()
      if len(out) == 0: continue
      rts = [float(p.get_selected_peakgroup().get_normalized_retentiontime()) for p in m.peptides.values() if p.get_selected_peakgroup() is not None]
      outlier_rts = [float(p.get_selected_peakgroup().get_normalized_retentiontime()) for p in m.peptides.values() if p.get_run_id() in out]
      mean_wo_outliers = numpy.mean([r for r in rts if r not in outlier_rts])
      for outlier_idx,outlier in enumerate(outlier_rts):
        if abs(outlier - mean_wo_outliers) > outlier_threshold_seconds:
            outliers += 1
            o.outlier_pg += 1
            thispep = [pep for pep in m.peptides.values() if pep.get_run_id() == out[outlier_idx]][0]
            newpg = thispep.find_closest_in_iRT(mean_wo_outliers)
            print "Bad", out, rts, "-> o", outlier_rts, "m", mean_wo_outliers, thispep.get_selected_peakgroup().get_normalized_retentiontime(), \
                " => exch", newpg.get_normalized_retentiontime(), newpg.get_fdr_score(), "/", thispep.get_selected_peakgroup().get_fdr_score()
            if( #abs(newpg.get_normalized_retentiontime() - best_rt_diff) < rt_diff_cutoff and 
                newpg.get_fdr_score() < aligned_fdr_cutoff):
                  thispep.get_selected_peakgroup().unselect_this_peakgroup()
                  newpg.select_this_peakgroup() 
                  o.outliers_changed += 1
                  print "change!"
        else: pass
    o.nr_outliers = outliers
    return o

def handle_args():
    import argparse

    usage = "" #usage: %prog --in \"files1 file2 file3 ...\" [options]" 
    usage += "\nThis program will select all peakgroups below the FDR cutoff in all files and try to align them to each other."
    usage += "\nIf only one file is given, it will act as peakgroup selector (best by m_score)" + \
            "\nand will apply the provided FDR cutoff."

    parser = argparse.ArgumentParser(description = usage )
    parser.add_argument('--in', dest="infiles", nargs = '+', help = 'A list of mProphet output files containing all peakgroups (use quotes around the filenames)')
    parser.add_argument("--out", dest="outfile", default="feature_alignment_outfile", help="Output file with filtered peakgroups for quantification")
    parser.add_argument("--fdr_cutoff", dest="fdr_cutoff", default=0.01, help="FDR cutoff to use, default 0.01", metavar='0.01', type=float)
    parser.add_argument("--max_rt_diff", dest="rt_diff_cutoff", default=30, help="Maximal difference in RT for two aligned features", metavar='30', type=float)
    parser.add_argument("--max_fdr_quality", dest="aligned_fdr_cutoff", default=0.2, help="Quality cutoff to still consider a feature for alignment (in FDR)", metavar='0.2', type=float)
    parser.add_argument("--outlier_thresh", dest="outlier_threshold_seconds", default=30, help="Everything below this threshold (in seconds), a peak will not be considered an outlier", metavar='30', type=float)
    parser.add_argument("--frac_selected", dest="min_frac_selected", default=0.0, help="Do not write peakgroup if selected in less than this fraction of runs (range 0 to 1)", metavar='0', type=float)
    parser.add_argument('--remove_outliers', action='store_true', default=False)
    parser.add_argument('--realign_runs', action='store_true', default=False, help="Tries to re-align runs based on their true RT (instead of using the less accurate iRT values by computing a spline against a reference run)")
    parser.add_argument('--use_scikit', action='store_true', default=False, help="Use datasmooth from scikit instead of R to re-align runs (needs to be installed)")
    parser.add_argument("--alignment_score", dest="alignment_score", default=0.0001, help="Minimal score needed for a feature to be considered for alignment between runs", metavar='0.0001', type=float)
    parser.add_argument('--method', default='best_overall', help="Which method to use for the clustering (best_overall or best_cluster_score)")
    parser.add_argument('--file_format', default='openswath', help="Which input file format is used (openswath or peakview)")

    args = parser.parse_args(sys.argv[1:])
    return args

def main(options):
    # Read the files
    this_exp = Experiment()
    #this_exp.parse_files(options.infiles, options.file_format, options.realign_runs)

    #import SWATHScoringReader
    reader = SWATHScoringReader.newReader(options.infiles, options.file_format)
    this_exp.runs = reader.parse_files(options.realign_runs)

    # Map the precursors across multiple runs, determine the number of
    # precursors in all runs without alignment.
    multipeptides = this_exp.get_all_multipeptides(options.fdr_cutoff)

    # If we want to align runs
    if options.realign_runs:
        this_exp.rt_align_all_runs(multipeptides, options.alignment_score, options.use_scikit)

    # do the alignment and annotate chromatograms without identified features
    # then perform an outlier detection over multiple runs
    alignment = align_features(multipeptides, options.rt_diff_cutoff, options.fdr_cutoff, options.aligned_fdr_cutoff, options.method)
    if options.remove_outliers:
      outlier_detection = detect_outliers(multipeptides, options.aligned_fdr_cutoff, options.outlier_threshold_seconds)
    else: outlier_detection = None

    # print statistics, write output
    this_exp.print_stats(multipeptides, alignment, outlier_detection, options.fdr_cutoff)
    this_exp.write_to_file(multipeptides, options.infiles, options.outfile, options.min_frac_selected, options.file_format)

if __name__=="__main__":
    options = handle_args()
    main(options)

