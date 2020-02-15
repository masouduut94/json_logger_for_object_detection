import unittest
import sys
from os import listdir
from time import sleep
from tempfile import mkdtemp
from unittest.case import TestCase

sys.path.append('../..')
from json_parser.json_parser import *


class TestFrame(TestCase):

    def setUp(self) -> None:
        self.json_parser = JsonParser()
        self.video_details = {'frame_width': 500,
                              'frame_height': 200,
                              'num_of_frames': 100,
                              'frame_rate': 20}
        self.out_path = mkdtemp()
        self.list_of_files_to_remove = []

    def test_automated_output_jsons(self):
        self.json_parser.set_top_k(0)
        self.json_parser.set_start()

        frames = [0, 1, 2]
        bbox_ids = [0, 1]
        bbox_xywh = [(10, 20, 30, 40), (50, 60, 70, 80)]
        time_to_output = 2
        start = datetime.now()

        for frame in frames:
            self.json_parser.add_frame(frame)
            sleep(1)
            for i in range(len(bbox_ids)):
                self.json_parser.add_bbox_to_frame(frame, bbox_ids[i], *bbox_xywh[i])
            self.json_parser.schedule_output(output_path=self.out_path, hours=0, minutes=0, seconds=time_to_output)
        end = datetime.now()
        saved_files = listdir(self.out_path)
        self.assertEqual(int((end - start).seconds / time_to_output), len(saved_files))


if __name__ == '__main__':
    unittest.main()
