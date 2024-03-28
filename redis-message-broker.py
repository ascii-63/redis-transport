import redis
import json
import sys


class RedisMessageBroker:
    def __init__(self, host='localhost', port=6379, db=0, queue=''):
        self.host = host
        self.port = port
        self.db = db
        self.queue = queue

        self.client = redis.Redis(host=self.host, port=self.port, db=self.db)

    def sendMessage(self, message: dict):
        self.client.lpush(self.queue, json.dumps(message))

    def receiveMessage(self):
        message = self.client.rpop(self.queue)
        if message:
            return str(message)
        return None


class Skeleton:
    def __init__(self, frame_id: int, person_id: int, bbox_: list,
                 kp: list,
                 kp_score: list):
        self.frame_id = frame_id
        self.person_id = person_id
        self.bbox = bbox_
        self.keypoints = kp
        self.keypoint_scores = kp_score


class Label:
    def __init__(self, label: str, bbox: list):
        self.label = label
        self.bbox = bbox


class DataCasting:
    sample_skeleton_dict = {"frame_id": 0, "person_id": 0,
                            "bbox": [], "keypoints": [], "keypoint_scores": []}
    sample_label_dict = {"label": "", "bbox": []}

    def __init__(self):
        pass

    def fromSkeletonDataToSkeletonDict(self, _frame_id: int, _person_id: int, _bbox: list, _kp: list, _kp_score: list) -> dict:
        if len(_bbox) == 4 and len(_kp) == len(_kp_score) == 17 and len(_kp[0]) == 2:
            ske_dict = self.sample_skeleton_dict

            ske_dict["frame_id"] = _frame_id
            ske_dict["person_id"] = _person_id
            ske_dict["bbox"] = _bbox
            ske_dict["keypoints"] = _kp
            ske_dict["keypoint_scores"] = _kp_score

            return ske_dict

        return None

    def fromSkeletonDictToSkeletonObject(_ske_dict) -> Skeleton:
        frame_id = _ske_dict["frame_id"]
        person_id = _ske_dict["person_id"]
        bbox = _ske_dict["bbox"]
        keypoints = _ske_dict["keypoints"]
        keypoint_scores = _ske_dict["keypoint_scores"]

        return Skeleton(frame_id, person_id, bbox, keypoints, keypoint_scores)

    def fromLabelDataToLabelDict(self, _label: str, _bbox: list) -> dict:
        if len(_bbox) == 4:
            label_dict = self.sample_label_dict

            label_dict["label"] = _label
            label_dict["bbox"] = _bbox

            return label_dict

        return None

    def fromLabelDictToLabelObject(_label_dict) -> Label:
        label = _label_dict["label"]
        bbox = _label_dict["bbox"]

        return Label(label, bbox)


class TransportAPI:
    def __init__(self, redis_host, redis_port, redis_db, skeleton_queue, label_queue):
        '''
        Crate a redis client to a redis server and set a queue
        '''
        self.redis_client1 = RedisMessageBroker(
            host=redis_host, port=redis_port, db=redis_db, queue=skeleton_queue)
        self.redis_client2 = RedisMessageBroker(
            host=redis_host, port=redis_port, db=redis_db, queue=label_queue)

    def sendSkeleton(self, frame_id: int, person_id: int, bbox: list, keypoints: list, keypoint_scores: list):
        '''
        Call this function to send a skeleton data.

        Parameters:
        ``frame_id`` (``int``): Frame ID
        ``person_id`` (``int``): Person ID in 'frame_id' frame
        ``bbox`` (``list``, size [4x1]): bounding box of person in frame
        ``keypoints`` (``list``, size [17x2]): keypoints of skeleton in 2D
        ``keypoint_scores`` (``list``, size [17x1]: score of each keypoint

        Return:
        ``bool``: Result of the sending request

        Raise:
        ``ValueError``: If the data is not in correct form/type
        '''

        skeleton_dict = DataCasting.fromSkeletonDataToSkeletonDict(DataCasting,
                                                                   frame_id, person_id, bbox, keypoints, keypoint_scores)
        if not skeleton_dict:
            raise ValueError("Data casting failed.")

        self.redis_client1.sendMessage(skeleton_dict)
        return True

    def getSkeleton(self):
        '''
        Call this function to get next skeleton object.

        Parameters:
        ``None``

        Return:
        ``Skeleton``: Skeleton data store in a ``Skeleton`` class object

        Raise:
        ``Exception``: An error occurred
        '''

        message = self.redis_client1.receiveMessage()
        if not message:
            raise Exception("Receive new message failed.")

        skeleton_dict = json.loads(message[2:-1])
        if not skeleton_dict:
            raise Exception("Json loads failed.")

        skeleton = DataCasting.fromSkeletonDictToSkeletonObject(skeleton_dict)
        if not skeleton:
            raise Exception("Data casting failed.")

        return skeleton

    def sendLabel(self, label: str, bbox: list):
        '''
        Call this function to send a label data.

        Parameters:
        ``label`` (``str``): Label
        ``bbox`` (``list``, size [4x1]): bounding box of person in frame

        Return:
        ``bool``: Result of the sending request

        Raise:
        ``ValueError``: If the data is not in correct form/type
        '''

        label_dict = DataCasting.fromLabelDataToLabelDict(
            DataCasting, label, bbox)
        if not label_dict:
            raise ValueError("Data casting failed.")

        self.redis_client2.sendMessage(label_dict)
        return True

    def getLabel(self):
        '''
        Call this function to get next label object.

        Parameters:
        ``None``

        Return:
        ``Label``: Label data store in a ``Label`` class object

        Raise:
        ``Exception``: An error occurred
        '''

        message = self.redis_client2.receiveMessage()
        if not message:
            raise Exception("Receive new message failed.")

        label_dict = json.loads(message[2:-1])
        if not label_dict:
            raise Exception("Json loads failed.")

        label = DataCasting.fromLabelDictToLabelObject(label_dict)
        if not label:
            raise Exception("Data casting failed.")

        return label


if __name__ == "__main__":  # DEMO
    # Create a Transport Object to call API
    transport = TransportAPI(redis_host='192.168.0.201', redis_port=6379, redis_db=0,
                             skeleton_queue='skeletons', label_queue='labels')

    # Create a sample skeleton data
    frame_id = 3
    person_id = 0
    bbox = [289, 1035, 174, 540]
    keypoints = [[1130, 341], [1142, 333], [1120, 332], [1158, 343], [1103, 342], [1172, 408], [1080, 406], [1185, 488], [
        1055, 483], [1176, 569], [1066, 561], [1153, 563], [1088, 562], [1163, 671], [1076, 676], [1171, 780], [1069, 788]]
    keypoint_scores = [0.98420066, 0.96603984, 0.9417951, 0.8331674, 0.70378006, 0.99427146, 0.9919647, 0.974518,
                       0.95928633, 0.95761883, 0.9389115, 0.9959512, 0.99517405, 0.98887664, 0.9855843, 0.9634876, 0.95550233]

    # Send skeleton data
    send_res = transport.sendSkeleton(frame_id=frame_id, person_id=person_id,
                                      bbox=bbox, keypoints=keypoints, keypoint_scores=keypoint_scores)
    if not send_res:
        sys.exit(1)

    # Get skeleton data
    skeleton_obj = transport.getSkeleton()
    if not skeleton_obj:
        sys.exit(2)
    print(skeleton_obj.keypoints)

    ###########################################

    # Create sample label, use above bbox
    label = 'Falling'

    # Send label data
    send_res = transport.sendLabel(label=label, bbox=bbox)
    if not send_res:
        sys.exit(3)

    # Get label data
    label_obj = transport.getLabel()
    if not label_obj:
        sys.exit(4)
    print(label_obj.label)

    sys.exit(0)
