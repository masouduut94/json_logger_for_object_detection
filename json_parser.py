import json
from os import makedirs
from os.path import exists, join
from bunch import bunchify
from datetime import datetime


class JsonMeta(object):
    HOURS = 3
    MINUTES = 59
    SECONDS = 59
    PATH_TO_SAVE = 'jsons'


class BaseJsonParser(object):
    """
    This is the base class that returns __dict__ of its own
    it also returns the dicts of objects in the attributes that are list instances

    """

    def dic(self):
        # returns dicts of objects
        out = {}
        for k, v in self.__dict__.items():
            if hasattr(v, 'dic'):
                out[k] = v.dic()
            elif isinstance(v, list):
                out[k] = self.list(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def list(values):
        # applies the dic method on items in the list
        return [v.dic() if hasattr(v, 'dic') else v for v in values]


class Label(BaseJsonParser):
    """
    For each bounding box there are various categories with confidences. Label class keeps track of that information.
    """

    def __init__(self, category: str, confidence: float):
        self.category = category
        self.confidence = confidence


class Bbox(BaseJsonParser):
    """
    This class keeps the information of each bounding box in the frame.

    """

    def __init__(self, bbox_id, top, left, width, height):
        self.labels = []
        self.bbox_id = bbox_id
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def add_label(self, category, confidence):
        # adds category and confidence only if top_k is not exceeded.
        self.labels.append(Label(category, confidence))

    def labels_full(self, value):
        return len(self.labels) == value

    def set_xywh_tuple(self, values: tuple):
        self.top, self.left, self.width, self.height = values

    def set_xywh_dict(self, values: dict):
        self.top = values['top']
        self.left = values['left']
        self.width = values['width']
        self.height = values['height']


class Frame(BaseJsonParser):
    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.bboxes = []

    def add_bbox(self, bbox_id: int, top: int, left: int, width: int, height: int):
        bboxes_ids = [bbox.bbox_id for bbox in self.bboxes]
        if bbox_id not in bboxes_ids:
            self.bboxes.append(Bbox(bbox_id, top, left, width, height))
        else:
            raise ValueError("Frame with id: {} already has a Bbox with id: {}".format(self.frame_id, bbox_id))


class JsonParser:
    def __init__(self, top_k_labels: int = 1):
        self.frames = {}
        self.video_details = dict(frame_width=None,
                                  frame_height=None,
                                  frame_rate=None,
                                  video_name=None)
        self.top_k_labels = top_k_labels
        self.start_time = datetime.now()

    def set_top_k(self, value):
        self.top_k_labels = value

    def frame_exists(self, frame_id: int):
        return frame_id in self.frames.keys()

    def add_frame(self, frame_id: int):
        # Use this function to add frames with index.
        if not self.frame_exists(frame_id):
            self.frames[frame_id] = Frame(frame_id)
        else:
            raise ValueError("Frame id: {} already exists".format(frame_id))

    def bbox_exists(self, frame_id, bbox_id):
        bboxes = []
        if self.frame_exists(frame_id=frame_id):
            bboxes = [bbox.bbox_id for bbox in self.frames[frame_id].bboxes]
        return bbox_id in bboxes

    def find_bbox(self, frame_id: int, bbox_id: int):
        if not self.bbox_exists(frame_id, bbox_id):
            raise ValueError("frame with id: {} does not contain bbox with id: {}".format(frame_id, bbox_id))
        bboxes = {bbox.bbox_id: bbox for bbox in self.frames[frame_id].bboxes}
        return bboxes.get(bbox_id)

    def add_bbox_to_frame(self, frame_id: int, bbox_id: int, top: int, left: int, width: int, height: int):
        if self.frame_exists(frame_id):
            frame = self.frames[frame_id]
            if not self.bbox_exists(frame_id, bbox_id):
                frame.add_bbox(bbox_id, top, left, width, height)
            else:
                raise ValueError(
                    "frame with frame_id: {} already contains the bbox with id: {} ".format(frame_id, bbox_id))
        else:
            raise ValueError("frame with frame_id: {} does not exist".format(frame_id))

    def add_label_to_bbox(self, frame_id: int, bbox_id: int, category: str, confidence: float):
        bbox = self.find_bbox(frame_id, bbox_id)
        if not bbox.labels_full(self.top_k_labels):
            bbox.add_label(category, confidence)
        else:
            raise ValueError("labels in frame_id: {}, bbox_id: {} is fulled".format(frame_id, bbox_id))

    def add_video_details(self, frame_width, frame_height, frame_rate, video_name):
        self.video_details['frame_width'] = frame_width
        self.video_details['frame_height'] = frame_height
        self.video_details['frame_rate'] = frame_rate
        self.video_details['video_name'] = video_name

    def output(self):
        output = {'video_details': self.video_details}
        result = list(self.frames.values())

        # Every bbox in each frame has to have `top_k_labels` number of labels otherwise error raises.
        for frame in result:
            for bbox in frame.bboxes:
                if not bbox.labels_full(self.top_k_labels):
                    raise ValueError(
                        "labels in frame_id: {}, bbox_id: {} is not fulled before outputting.".format(frame.frame_id,
                                                                                                      bbox.bbox_id))

        output['frames'] = [item.dic() for item in result]
        return output

    def object_output(self):
        output = bunchify(self)
        return output

    def json_output(self, output_name):
        if not output_name.endswith('.json'):
            output_name += '.json'
        with open(output_name, 'w') as file:
            json.dump(self.output(), file)

    def set_start(self):
        self.start_time = datetime.now()

    def schedule_output(self, output_path=JsonMeta.PATH_TO_SAVE, hours: int = 0, minutes: int = 0, seconds: int = 60):
        end = datetime.now()
        interval = 0
        interval += min([hours, JsonMeta.HOURS]) * 3600
        interval += min([minutes, JsonMeta.MINUTES]) * 60
        interval += min([seconds, JsonMeta.SECONDS])
        diff = (end - self.start_time).seconds

        if diff > interval:
            output_name = self.start_time.strftime('%Y-%m-%d %H-%M-%S') + '.json'
            output = join(output_path, output_name)
            if not exists(output_path):
                makedirs(output_path)
            self.json_output(output_name=output)
            self.frames.clear()
            self.start_time = datetime.now()


if __name__ == '__main__':
    frames_ids = [1, 2, 3, 4, 5]
    bbox_ids = [0, 1, 2, 3]
    bbox_xywh = [(1, 5, 200, 100), (5, 50, 250, 400), (10, 21, 100, 210), (20, 30, 10, 200)]
    top_k = 3
    label_category = ['a', 'b', 'c']
    label_conf = [98, 93, 25]

    json_parser = JsonParser(top_k_labels=top_k)
    i = 0

    while i < len(frames_ids):
        json_parser.add_frame(frames_ids[i])
        for bbox_id, bbox_xywh_item in zip(bbox_ids, bbox_xywh):
            x, y, w, h = bbox_xywh_item
            json_parser.add_bbox_to_frame(frame_id=frames_ids[i], bbox_id=bbox_id, top=x, left=y, width=w, height=h)
            for category, confidence in zip(label_category, label_conf):
                json_parser.add_label_to_bbox(frame_id=frames_ids[i], bbox_id=bbox_id, category=category,
                                              confidence=confidence)
        i += 1
    video_details = {'frame_width': 1200, 'frame_height': 720, 'frame_rate': 20}
    json_parser.add_video_details(**video_details)
    path = 'camera_1'
    if not exists(path):
        makedirs(path)
    output_name = join(path, 'json.json')
    out = json_parser.output()
    json_parser.json_output(output_name=output_name)
    # d = bunchify(json_parser)
    # print(d.frames[1].bboxes[0].labels[0].category)
