#!/usr/bin/python
# -*- coding: utf-8 -*-
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

import os
import pymzml
from SingleChromatogramFile import SingleChromatogramFile

class SwathRun(object):
    """Data model for an individual SWATH injection (may contain multiple mzML files).

    This contains the model for all data from a single run (e.g. one panel in
    the viewer - in reality this could be multiple actual MS runs since in SRM
    not all peptides can be measured in the same injection or just multiple
    files generated by SWATH MS.

    It abstracts all the interfaces of SingleChromatogramFile, usually all
    other classes directly communicate with this class.

    Attributes:
        runid: Current run id

    Private Attributes:
        _all_swathes:        Dictionary of { mz : SingleChromatogramFile }

        _files:              List of files that are containing data for this run

        _in_memory:          Whether data should be held in memory

        TODO: the following three attributes are set by _read_peakgroup_files violating encapsulation)

        _range_mapping:      Dictionary of { precursor_id : [leftWidth, rightWidth] }

        _score_mapping:      Dictionary of { precursor_id : FDR_score }

        _intensity_mapping:  Dictionary of { precursor_id : Intensity }

    """

    def __init__(self, files, runid=None, precursor_mapping = None, sequences_mapping = None):
        self.runid = runid
        self._all_swathes = {}
        self._in_memory = False
        self._files = files

        # extra info
        self._range_mapping = {}
        self._score_mapping = {}
        self._intensity_mapping = {}

        self._loadFiles(files, precursor_mapping, sequences_mapping)
        self._initialize()

    def _loadFiles(self, files, precursor_mapping = None, sequences_mapping = None):
        """
        Load the files associated with this run using pymzml

        Each run is stored in the _all_swathes dictionary where the runs are
        accessible through the m/z of the first precursor.
        """
        for f in files:
            print "Loading file", f
            import time
            start = time.time()
            run_ = pymzml.run.Reader(f, build_index_from_scratch=True)
            print "Loading file", f, "took", time.time() - start
            run_.original_file = f
            first = run_.next()
            mz = first['precursors'][0]['mz']
            self._all_swathes[ int(mz) ] = SingleChromatogramFile(run_, f, 
                precursor_mapping=precursor_mapping, sequences_mapping=sequences_mapping)

    def _initialize(self):
        """ 
        Map the individual sequences and precursors to the respective swath file
        """

        self._precursor_run_map = {}
        self._sequences_mapping = {}

        for run_key, run in self._all_swathes.iteritems():

            for key in run._precursor_mapping:
                self._precursor_run_map[key] = run_key

            for key in run._sequences_mapping:
                tmp = self._sequences_mapping.get(key, [])
                tmp.extend( run._sequences_mapping[key] )
                self._sequences_mapping[key] = tmp

    #
    ## Getters (info)
    #
    def get_precursors_for_sequence(self, sequence):
        return self._sequences_mapping.get(sequence, [])

    def get_transitions_for_precursor(self, precursor):
        run = self._precursor_run_map.get( str(precursor), None)
        if run is None:
            return []
        return self._all_swathes[run].get_transitions_for_precursor(precursor)

    def get_transitions_for_precursor_display(self, precursor):
        run = self._precursor_run_map.get( str(precursor), None)
        if run is None:
            return []
        return self._all_swathes[run].get_transitions_with_mass_for_precursor(precursor)

    def get_all_precursor_ids(self):
        return self._precursor_run_map.keys()

    def get_all_peptide_sequences(self):
        res = set([])
        for m in self._all_swathes.values():
            res.update( m._sequences_mapping.keys() )
        return res

    #
    ## Getters (data) -> see ChromatogramTransition.getData
    #
    # Some of these functions are just aggregation functions over all
    # individual .chrom.mzML files (e.g. selecting the correct run and then
    # getting the raw data from it or summing over all runs).
    #
    def getTransitionCount(self):
        """
        Aggregate transition count over all files
        """
        return sum([r.getTransitionCount() for r in self._all_swathes.values()] )

    def get_data_for_precursor(self, precursor):
        """
        Retrieve raw data for a specific precursor (using the correct run).
        """

        run = self._precursor_run_map[str(precursor)]
        return self._all_swathes[run].get_data_for_precursor(precursor)

    def get_data_for_transition(self, transition_id):
        """
        Retrieve raw data for a specific transition (using the correct run).
        """

        for run in self._all_swathes.values():
            if len( run.get_data_for_transition(transition_id)[0][0] ) > 1:
                return run.get_data_for_transition(transition_id)

        # Default value
        return run.get_data_for_transition(transition_id)

    def get_range_data(self, precursor):
        return self._range_mapping.get(precursor, [0,0])

    def get_score_data(self, precursor):
        return self._score_mapping.get(precursor, None)

    def get_intensity_data(self, precursor):
        return self._intensity_mapping.get(precursor, None)

    def get_id(self):
        fileid = ""
        if len(self._files) > 0:
            fileid = os.path.basename(self._files[0]) 

        return self.runid + "_" + fileid

    # 
    ## Data manipulation
    #
    def remove_precursors(self, toremove):
        """ Remove a set of precursors from the run (this can be done to filter
        down the list of precursors to display).
        """
        for run_key, run in self._all_swathes.iteritems():
            for key in toremove:
                run._precursor_mapping.pop(key, None)
                self._precursor_run_map.pop(key, None)
            run._group_precursors_by_sequence()

        # Re-initialize self to produce correct mapping
        self._initialize()

