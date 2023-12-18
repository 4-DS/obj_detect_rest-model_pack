from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, FileInput
from bentoml.service.artifacts.common import TextFileArtifact, JSONArtifact

from sinara.bentoml_artifacts import OnnxModelArtifact, BinaryFileArtifact

from typing import List, BinaryIO
import io
import json
import numpy as np
import cv2
from PIL import Image


@env(infer_pip_packages=True)
@artifacts([
    OnnxModelArtifact('model', backend='onnxruntime'),
    BinaryFileArtifact('test_image', file_extension='.jpg'),
    BinaryFileArtifact('test_result', file_extension=".pkl"),
    TextFileArtifact('service_version',
                             file_extension='.txt',
                             encoding='utf8')]) # for versions of bentoml 0.13 and newer 

class ModelService(BentoService): 
    def __init__(self):
        super().__init__() 
        self.pre_post_processing = PrePostProcessing()
        
    @api(input=JsonInput(), batch=False)
    def service_version(self, *args): 
        """ Return version of a running service """
        return self.artifacts.service_version
    
    @api(input=JsonInput())
    def test_data(self, *args): # return some data for running a test
        return self.artifacts.test_image
    
    @api(input=JsonInput(), batch=False)
    def test_result(self, *args): # return result data for a test data
        return self.artifacts.test_result    
    
    @api(input=FileInput(), batch=False)
    def predict(self, file_stream):
        input_data, scale_factors = self.pre_post_processing.prep_processing(file_stream)
        input_name = self.artifacts.model.get_inputs()[0].name
        output_name = [out.name for out in self.artifacts.model.get_outputs()]
        outs = self.artifacts.model.run(output_name, {input_name: input_data})        
        outs = self.pre_post_processing.post_processing(outs, scale_factors)
        return outs
        
        
class PrePostProcessing:
    def __init__(self):
        self.input_size = (640, 640)
    
    def prep_processing(self, file_stream):        
        pil_img=Image.open(file_stream)
        image_array = np.asarray(pil_img)
        image_array = image_array[...,::-1] # to bgr
        resized = cv2.resize(image_array, self.input_size).astype(np.float32)
        scale_y, scale_x = self.input_size[0]/image_array.shape[0], self.input_size[1]/image_array.shape[1]
        scale_factors = [scale_x, scale_y]*2
        input_data = resized.transpose(2,0,1)[np.newaxis, ...].astype(np.float32)
        return input_data, scale_factors
    
    def post_processing(self, output_data, scale_factors):        
        scale_x, scale_y = scale_factors[:2]
        dets, labels = output_data
        dets = dets/np.array(scale_factors+[1.0])
        return dets, labels 
