'''

tests for tokens module

'''
import unittest
import pandas as pd
from collections import Counter

from processor import Processor
from doc import DocManager


import settings


class ProcessorTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ProcessorTest, self).__init__(*args, **kwargs)

        self.processor = Processor(
            chars_to_remove = settings.NPL_CHARS_TO_REMOVE
        )
    
    def test_preprocessor(self):
        file_content = 'a-bb ccc+dd\neee? ff'
        content_processed = 'a bb ccc dd eee ff'
        self.assertEqual(
            self.processor.process(file_content),
            content_processed
        )

    def test_filter(self):

        '''

        Pure code

        Segment type:	Externs


        '''
    


# class DocTest(unittest.TestCase):

#     def __init__(self, *args, **kwargs):
#         super(DocTest, self).__init__(*args, **kwargs)

#         self.file_content = 'a-bb ccc+dd\neee? ff'
#         with open(f'{settings.TEST_PATH_DATA_RAW}/a.asm', 'w+') as f:
#             f.write(self.file_content)

#         self.processor = Processor(
#             chars_to_remove = settings.NPL_CHARS_TO_REMOVE
#         )

#         self.doc_manager = DocManager(
#             docs_limit = 10,
#             path_data_raw = settings.TEST_PATH_DATA_RAW,
#             path_data_counts = settings.TEST_PATH_DATA_COUNTS,
#             processor = self.processor,
#             max_features = 1000,
#             ngram = 2,
#             max_workers = 1
#         )

    

#     def test_create_count_from_single_asm(self):
#         values = (1, 1, 'a.asm')

#         # ngram = 1
#         self.doc_manager.set_ngram(1)
#         vocab_1 = self.doc_manager.create_count_from_single_asm(values)
#         counter_1 = Counter({'a': 1, 'bb': 1, 'ccc': 1, 'dd': 1, 'eee': 1, 'ff': 1})
#         self.assertEqual(vocab_1, counter_1)

#         # ngram = 2
#         self.doc_manager.set_ngram(2)
#         vocab_2 = self.doc_manager.create_count_from_single_asm(values)
#         counter_2 = Counter({'a bb': 1, 'bb ccc': 1, 'ccc dd': 1, 'dd eee': 1, 'eee ff': 1})
#         self.assertEqual(vocab_2, counter_2)

#     def test_accumulate_counts(self):
#         values = [
#             (1, 2, 'a.txt'),
#             (2, 2, 'a.txt')
#         ]

#         # ngram = 1
#         self.doc_manager.set_ngram(1)
#         acm_values_1 = Counter({'a': 2, 'bb': 2, 'ccc': 2, 'dd': 2, 'eee': 2, 'ff': 2})
#         self.assertEqual(
#             self.doc_manager.accumulate_counts(values),
#             acm_values_1
#         )

#         # ngram = 2
#         self.doc_manager.set_ngram(2)
#         acm_values_2 = Counter({'a bb': 2, 'bb ccc': 2, 'ccc dd': 2, 'dd eee': 2, 'eee ff': 2})
#         self.assertEqual(
#             self.doc_manager.accumulate_counts(values),
#             acm_values_2
#         )
        



def main():
    unittest.main()

if __name__ == '__main__':
    main()