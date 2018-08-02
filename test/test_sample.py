import unittest
import random
import sys

sys.path.append('../')
import ollin


class TestSample(unittest.TestCase):
    def test_sample(self):
        self.assertTrue(True)

    def test_site_creation(self):
        niche_size = random.random()
        site = ollin.Site.make_random(niche_size)

        self.assertTrue(abs(site.niche_size - niche_size) < 0.1)
