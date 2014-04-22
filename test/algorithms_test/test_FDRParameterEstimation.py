#!/usr/bin/python
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

import os

import unittest
from nose.plugins.attrib import attr

import msproteomicstoolslib.algorithms.alignment.FDRParameterEstimation as estimator

class MockRun():

    def __init__(self, id_):
        self.id_ = id_
        self.orig_filename = "test"

    def get_id(self):
        return self.id_


def prepareData(nr_peps=15):

    import msproteomicstoolslib.data_structures.Precursor as precursor
    import msproteomicstoolslib.format.TransformationCollection as transformations
    from msproteomicstoolslib.algorithms.alignment.SplineAligner import SplineAligner
    import msproteomicstoolslib.algorithms.alignment.AlignmentHelper as helper
    from msproteomicstoolslib.algorithms.alignment.Multipeptide import Multipeptide

    # 0. id
    # 1. quality score (FDR)
    # 2. retention time (normalized)
    # 3. intensity

    multipeptides = []

    runs = [MockRun("0_%s" % (i+1)) for i in range(5)]
    ids = 0
    for j in range(1,nr_peps):

        mpeps = [Multipeptide() for i in range(2)]
        [m.set_nr_runs(5) for m in mpeps]
        # Iterate over all runs
        for i in range(5):
            # A target peptide
            p = precursor.Precursor("target_pep_%s" % j, runs[i] )
            p.set_decoy("FALSE")
            pg_tuple = ("id_%s" % ids, 1.0/j, 100, 10000)
            p.add_peakgroup_tpl(pg_tuple, "target_pep_%s" % j, -1)
            mpeps[0].insert(runs[i].get_id(), p)
            ids += 1

            # A decoy peptide
            p = precursor.Precursor("decoy_pep_%s" % j, runs[i] )
            p.set_decoy("TRUE")
            pg_tuple = ("id_%s" % ids, 1.0/j, 100, 10000)
            p.add_peakgroup_tpl(pg_tuple, "decoy_pep_%s" % j, -1)
            mpeps[1].insert(runs[i].get_id(), p)
            ids += 1
        multipeptides.extend(mpeps)


    # Add some more target peptides at good FDRs 
    for j in range(1,nr_peps):

        mpeps = [Multipeptide() for i in range(1)]
        [m.set_nr_runs(5) for m in mpeps]
        # Iterate over all runs
        for i in range(5):
            p = precursor.Precursor("target_pep_good_%s" % j, runs[i] )
            p.set_decoy("FALSE")
            pg_tuple = ("id_%s" % ids, 0.01/j, 100, 10000)
            p.add_peakgroup_tpl(pg_tuple, "target_pep_good_%s" % j, -1)
            mpeps[0].insert(runs[i].get_id(), p)
            ids += 1
        multipeptides.extend(mpeps)
    return multipeptides

class TestUnitAlignmentParamEstimator(unittest.TestCase):

    def setUp(self):
        pass

    def test_creatObject(self):
        e = estimator.ParamEst()
        self.assertTrue(True)

    def test_findFDR(self):
        multipeptides = prepareData(15)
        e = estimator.ParamEst(verbose=False)
        res = e.find_iterate_fdr(multipeptides, 0.10)
        self.assertAlmostEqual(res, 0.0788571428571)

        res = e.find_iterate_fdr(multipeptides, 0.20)
        self.assertAlmostEqual(res, 0.103428571429)

        res = e.find_iterate_fdr(multipeptides, 0.30)
        self.assertAlmostEqual(res, 0.252571428571)

    def test_tooLargeFDR(self):

        multipeptides = prepareData(15)
        e = estimator.ParamEst(verbose=False)
        self.assertRaises(Exception, e.find_iterate_fdr,multipeptides, 0.80)

if __name__ == '__main__':
    unittest.main()
