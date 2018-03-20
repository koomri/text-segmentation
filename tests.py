from __future__ import print_function

from unittest import TestCase
from utils import unsort
import unittest
import accuracy
import numpy as np
import text_manipulation

class LoaderTests(TestCase):
    def testReallyTrivial(self):
        assert 1 + 1 == 2

class PkTests(unittest.TestCase):
    def test_get_boundaries(self):
        sentences_class = []
        sentences_class.append(("first sen.", 1))
        sentences_class.append(("sec sen.", 1))
        sentences_class.append(("third sen.", 0))
        sentences_class.append(("forth sen.", 1))
        sentences_class.append(("fifth sen.", 0))
        sentences_class.append(("sixth sen.", 0))
        sentences_class.append(("seventh sen.", 1))

        expected = [2, 2, 4, 6]
        result = accuracy.get_seg_boundaries(sentences_class)

        for i, num in enumerate(result):
            self.assertTrue(num == expected[i])

    def test_get_boundaries2(self):
        sentences_class = []
        sentences_class.append(("first sen is 5 words.", 0))
        sentences_class.append(("sec sen.", 0))
        sentences_class.append(("third sen is a very very very long sentence.", 1))
        sentences_class.append(("the forth one is single segment.", 1))


        expected = [16, 6]
        result = accuracy.get_seg_boundaries(sentences_class)

        for i, num in enumerate(result):
            self.assertTrue(num == expected[i])

    def test_pk_perefct_seg(self):
        sentences_class = []
        sentences_class.append(("first sen is 5 words.", 0))
        sentences_class.append(("sec sen.", 0))
        sentences_class.append(("third sen is a very very very long sentence.", 1))
        sentences_class.append(("the forth one is single segment.", 1))

        gold = accuracy.get_seg_boundaries(sentences_class)
        h = accuracy.get_seg_boundaries(sentences_class)

        # with specified window size
        for window_size in range(1, 15, 1):
            acc = accuracy.pk(gold, h, window_size=window_size)
            self.assertEquals(acc, 1)

        # with default window size
        acc = accuracy.pk(gold, h)
        self.assertEquals(acc, 1)

    def test_pk_false_neg(self):
        h = []
        h.append(("5 words sentence of data.", 0))
        h.append(("2 sentences same seg.", 1))

        gold = []
        gold.append(("5 words sentence of data.", 1))
        gold.append(("2 sentences same seg.", 1))


        gold = accuracy.get_seg_boundaries(gold)
        h = accuracy.get_seg_boundaries(h)

        window_size = 3
        comparison_count = 6

        # with default window size
        acc = accuracy.pk(gold, h)
        self.assertEquals(acc, window_size / comparison_count)

        window_size = 4
        acc = accuracy.pk(gold, h)
        self.assertEquals(acc, window_size / comparison_count)

    def test_windiff(self):
        h = []
        h.append(("5 words sentence of data.", 0))
        h.append(("short.", 1))
        h.append(("extra segmented sen.", 1))
        h.append(("last and very very very very very long sen.", 1))


        gold = []
        gold.append(("5 words sentence of data.", 1))
        gold.append(("short.", 1))
        gold.append(("extra segmented sen.", 0))
        gold.append(("last and very very very very very long sen.", 1))


        gold = accuracy.get_seg_boundaries(gold)
        h = accuracy.get_seg_boundaries(h)

        window_size = 3

        acc = accuracy.win_diff(gold, h, window_size = window_size)
        self.assertEquals(float(acc), 0.6)
        
        window_size = 5
        expected = float(1)- float(8) / 13

        acc = accuracy.win_diff(gold, h, window_size=window_size)
        self.assertEquals("{0:.5f}".format(float(acc)), "{0:.5f}".format(expected))


class UnsortTests(TestCase):
    def test_unsort(self):
        x = np.random.randint(0, 100, 10)
        sort_order = np.argsort(x)
        unsort_order = unsort(sort_order)
        assert np.all(x[sort_order][unsort_order] == x)


class SentenceTokenizerTests(TestCase):
    def test_a_little(self):
        a = text_manipulation.split_sentences(u"Hello, Mr. Trump, how do you do? What? Where? I don't i.e e.g Russia.")
        assert a == [u'Hello, Mr. Trump, how do you do?',
                     u'What?',
                     u'Where?',
                     u"I don't i.e e.g Russia."]

    def test_linebreaks(self):
        text = u'''Line one. Still line one.

        Line two. Can I span
        two lines?'''
        a = text_manipulation.split_sentences(text)
        print(a)
        assert a == [u'Line one.',
                     u'Still line one.',
                     u'Line two.',
                     u'Can I span\n        two lines?']





if __name__ == '__main__':
    unittest.main()