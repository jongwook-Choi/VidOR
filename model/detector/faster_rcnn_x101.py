from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures.boxes import Boxes
import torch


class FasterRCNNX101:
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            "./configs/COCO-Detection/custom_faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
        cfg.MODEL.DEVICE = 'cuda'
        cfg.MODEL.WEIGHTS = './weight/pretrained_weight/faster_rcnn/model_0159999.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.predictor = DefaultPredictor(cfg)
        # self.model 원본 이미지에서 feature를 추출하기 위해 사용
        self.model = self.predictor.model

    '''
    x0 좌표 y0 좌표 x1 좌표 y1 좌표 (x1 > x0, y1 > y0) 두개
    boxes =
        tensor([[ 41.5078,   0.0000, 475.0139, 357.1633],
        [ 83.9284,  75.0421, 333.0050, 355.9797]], device='cuda:0')
    (x0, y0, x1, y1), mode='xyxy'
    
    30번 28번 2개
    classes = 
        tensor([30, 28], device='cuda:0')
    
    신뢰점수 2개
    scores = 
        tensor([0.9998, 0.9984], device='cuda:0')
    '''

    def detect(self, img):
        outputs = self.predictor(img)
        boxes = outputs["instances"].pred_boxes.tensor
        classes = outputs["instances"].pred_classes
        scores = outputs["instances"].scores

        return boxes, classes, scores

    def feature_extraction(self, img, boxes):
        """
        :param img: numpy.ndarray(cv2.imread)
        :param boxes: [N X 4] torch.tensor(N = bbox num, 4 = bbox coordinate, dtype = cuda)
        :return: box_features: [N X 1024] torch.tensor(N = bbox num, 1024 = feature dimension, dtype = torch.float32)
        """
        height, width = img.shape[:2]
        image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]

        boxes = boxes.to('cpu')
        boxes_instance = Boxes(torch.Tensor(boxes))
        boxes_instance = boxes_instance.to('cuda')

        with torch.no_grad():
            images = self.model.preprocess_image(inputs)
            features = self.model.backbone(images.tensor)
            features_ = [features[f] for f in self.model.roi_heads.box_in_features]

            gt_boxes = [boxes_instance]
            box_features = self.model.roi_heads.box_pooler(features_, gt_boxes)
            box_features = self.model.roi_heads.box_head(box_features)

            box_features = box_features.to('cpu')

        return box_features

