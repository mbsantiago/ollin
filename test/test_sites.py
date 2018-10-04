import unittest
import numpy as np

import sys
sys.path.append('../')
import ollin  # noqa: E402


class TestSites(unittest.TestCase):
    def test_range_inputs(self):
        random_niche = np.random.random(size=(20, 20))

        # Test range by integer
        r = 10
        site = ollin.BaseSite(r, random_niche)
        self.assertTrue((site.range == np.array([float(r), float(r)])).all())
        self.assertTrue(site.range.dtype == np.float)

        # Test range by float
        r = 15.0
        site = ollin.BaseSite(r, random_niche)
        self.assertTrue((site.range == np.array([r, r])).all())
        self.assertTrue(site.range.dtype == np.float)

        # Test range by tuple
        r = (10, 20.0)
        site = ollin.BaseSite(r, random_niche)
        self.assertTrue((site.range == np.array(r)).all())
        self.assertTrue(site.range.dtype == np.float)

        # Test range by list
        r = [10, 20.0]
        site = ollin.BaseSite(r, random_niche)
        self.assertTrue((site.range == np.array(r)).all())
        self.assertTrue(site.range.dtype == np.float)

        # Test range by array
        r = np.array([10.0, 20.0])
        site = ollin.BaseSite(r, random_niche)
        self.assertTrue((site.range == r).all())
        self.assertTrue(site.range.dtype == np.float)
