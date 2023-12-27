from typing import List, BinaryIO
import io
import json
import numpy as np
import cv2
from PIL import Image


class PrePostProcessing:
    
    def prep_processing(self, file_stream, input_size=(640,640)):     
        pil_img=Image.open(file_stream)
        image_array = np.asarray(pil_img)
        image_array = image_array[...,::-1] # to bgr
        img_ori_size = image_array.shape[:2]
        resized = cv2.resize(image_array, input_size).astype(np.float32)
        
        scale_y, scale_x = input_size[0]/img_ori_size[0], input_size[1]/img_ori_size[1]
        scale_factors = [scale_x, scale_y]*2
        processing_img = resized.transpose(2,0,1)[np.newaxis, ...].astype(np.float32)
        return processing_img, scale_factors, img_ori_size
    
    def post_processing(self, predicts, scale_factors, img_ori_size, categories):        
        scale_x, scale_y = scale_factors[:2]
        dets, labels = predicts
        dets = dets/np.array(scale_factors+[1.0])
        image_meta = {"file_name": "unknown.jpg", "id": 1, "img_size":  img_ori_size}
        coco_predict = self.convert_inference_results_to_coco(dets[0], labels[0], image_meta)    
        coco_predict["categories"] = categories
        return coco_predict 
    
    @staticmethod
    def convert_inference_results_to_coco(dets, labels, image_meta, min_score: float = 0) -> dict:
        selected_classes = []
        selected_boxes = []
        selected_scores = []
        selected_cnts = []

        if labels.size > 0:
            selected_scores = dets[:,4]
            selected_boxes = dets[:,:4]
            selected_classes = labels
            # print(selected_classes)
            # break

        box_to_cnt_indexes = [[0, 1], [2, 1], [2, 3], [0, 3], [0, 1]]
        for bbox in selected_boxes:
            cnt = []
            for a, b in box_to_cnt_indexes:
                cnt.append([bbox[a], bbox[b]])
            selected_cnts.append([cnt])
        
        img_size = image_meta["img_size"]
        image_id = image_meta['id']        
        """
            Optimized universal predictor
            :param selected_classes - numpy array of classes (-1,1)
            :param selected_cnts    - numpy array of contours (-1,-1,1,2)
            :param selected_boxes   - numpy array bbox (-1,4)
            :param selected_scores   - numpy array bbox (-1,1)
            :param output_shape     - output image shape. Usung only 2 first values
            :param file_name        - input file name
            :param use_segm         - using segmentation model or object detection only
            :param image_id         - input file id
            :param min_score        - filtering condition

            :return: dict
        """

        annotations = [{
            'id': i,
            'base_id': i,
            'image_id': int(image_id),
            # bbox x1,y1,w,h describing the object
            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
            # Object category
            #'category_id': loaded_categories_remap[int(selected_classes[i])],
            'category_id': int(selected_classes[i])+1,
            'score': float(selected_scores[i]),  # confidence
            # Contour array x,y,x,y
            'segmentation': [np.int0(_cnt).ravel().tolist() for _cnt in selected_cnts[i]],
        } for i, ((x1, y1, x2, y2), score) in enumerate(zip(selected_boxes,selected_scores)) if score > min_score]

        for i in range(len(annotations)):
            del annotations[i]['base_id']

        return {
            'file_name': image_meta['file_name'],  # not used
            'height': img_size[0],  # Image Resolution
            'width': img_size[1],  # Image Resolution
            'image_id': int(image_id),
            #'CLASSES_COUNT': len(loaded_categories),
            'annotations': annotations
        }
    