import unittest
import sys
from unittest.case import TestCase

sys.path.append('../..')
from json_parser.json_parser import *


class TestFrame(TestCase):

    def setUp(self) -> None:
        self.json_parser = JsonParser()
        self.video_details = {'frame_width': 500,
                              'frame_height': 200,
                              'frame_rate': 20,
                              'video_name': 'something.mp4'}

    def test_add_frame(self):
        frames_ids = [1, 2, 3, 4, 5]
        self.json_parser.set_top_k(0)
        for frame_id in frames_ids:
            self.json_parser.add_frame(frame_id)
        output = self.json_parser.output()
        output_ids = [frame['frame_id'] for frame in output['frames']]
        for frame_id in frames_ids:
            self.assertIn(frame_id, output_ids)

        # if repeated frame id was inserted, raise ValueError
        with self.assertRaisesRegex(ValueError, "Frame id: (.*?) already exists"):
            self.json_parser.add_frame(frames_ids[1])

    def test_add_bbox_to_frame(self):
        # test if bbox is added to its own frame
        frame_one_id = 0
        bbox_id = 1
        bbox_xywh = (10, 20, 30, 40)
        top_k = 0
        self.json_parser.set_top_k(top_k)

        self.json_parser.add_frame(frame_one_id)
        self.json_parser.add_bbox_to_frame(frame_one_id, bbox_id, *bbox_xywh)
        out = self.json_parser.output()
        # retrieve the bbox in the frame
        inserted_bbox = out['frames'][0]['bboxes'][0]
        # check the values inserted in bbox
        self.assertEqual((inserted_bbox['bbox_id']), bbox_id)
        self.assertEqual(inserted_bbox['top'], bbox_xywh[0])
        self.assertEqual(inserted_bbox['left'], bbox_xywh[1])
        self.assertEqual(inserted_bbox['width'], bbox_xywh[2])
        self.assertEqual(inserted_bbox['height'], bbox_xywh[3])

        # check if error raises with repeated insertion
        with self.assertRaisesRegex(ValueError,
                                    'frame with frame_id: (.*?) already contains the bbox with id: (.*?) '):
            self.json_parser.add_bbox_to_frame(frame_one_id, bbox_id, *bbox_xywh)

        # check if error raises when frame_id is not recognized
        with self.assertRaisesRegex(ValueError,
                                    'frame with frame_id: (.*?) does not exist'):
            self.json_parser.add_bbox_to_frame(2, bbox_id, *bbox_xywh)

    def test_add_label_to_bbox(self):
        # test if label exists in bbox
        frame_one_id = 0
        bbox_id = 1
        bbox_xywh = (10, 20, 30, 40)
        self.json_parser.add_frame(frame_one_id)
        self.json_parser.add_bbox_to_frame(frame_one_id, bbox_id, *bbox_xywh)

        categories = ['car', 'truck', 'bus']
        confidences = [0.47, 0.85, 1]
        top_k = 3

        self.json_parser.set_top_k(value=top_k)
        for cat, conf in zip(categories, confidences):
            self.json_parser.add_label_to_bbox(frame_one_id, bbox_id, cat, conf)
        out = self.json_parser.output()
        inserted_labels = out['frames'][0]['bboxes'][0]['labels']

        # Does output contain all labels that were inserted.
        for label in inserted_labels:
            self.assertTrue(label['category'] in categories)
            self.assertTrue(label['confidence'] in confidences)

        # Check if appending extra label (more than top_k) raises ValueError
        with self.assertRaisesRegex(ValueError, 'labels in frame_id: (.*?), bbox_id: (.*?) is fulled'):
            self.json_parser.add_label_to_bbox(frame_one_id, bbox_id, 'new_category', 0.12)

        # Check if error raises when the number of inserted labels are less than top_k
        self.json_parser.set_top_k(4)
        with self.assertRaisesRegex(ValueError,
                                    'labels in frame_id: (.*?), bbox_id: (.*?) is not fulled before outputting.'):
            out = self.json_parser.output()


if __name__ == '__main__':
    unittest.main()
