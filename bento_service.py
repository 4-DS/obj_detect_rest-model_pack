from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, FileInput
from bentoml.service.artifacts.common import TextFileArtifact, JSONArtifact

from pre_post_processing import PrePostProcessing

# All code needed for SinaraML BentoService profiles is managed by SinaraML team
from sinara.bentoservice import *


pre_post_processing = PrePostProcessing()

@env(infer_pip_packages=True)
@artifacts([
    OnnxModelArtifact('model', backend='onnxruntime-gpu'),
    BinaryFileArtifact('test_image', file_extension='.jpg'),
    BinaryFileArtifact('test_result', file_extension=".pkl"),
    TextFileArtifact('service_version',
                             file_extension='.txt',
                             encoding='utf8'),
    JSONArtifact("categories"),
    JSONArtifact("input_size")
    
    ]) # for versions of bentoml 0.13 and newer
@SinaraOnnxBentoService()
class ModelService(BentoService): 
    
    @api(input=JsonInput())
    def test_data(self, *args): # return some data for running a test
        return self.artifacts.test_image
    
    @api(input=JsonInput(), batch=False)
    def test_result(self, *args): # return result data for a test data
        return self.artifacts.test_result    
    
    @api(input=FileInput(), batch=False)
    def predict(self, file_stream):
        # pre_post_processing = PrePostProcessing(categories = self.artifacts.categories,
        #                                         input_size = self.artifacts.input_size)
        
        # preprocessing image
        processing_img, scale_factors, img_ori_size = pre_post_processing.prep_processing(file_stream, 
                                                                                          input_size = self.artifacts.input_size)
        
        # inference onnx 
        input_name = self.artifacts.model.get_inputs()[0].name
        output_name = [out.name for out in self.artifacts.model.get_outputs()]
        predicts = self.artifacts.model.run(output_name, {input_name: processing_img})  
        
        # postprocessing predicts
        predicts = pre_post_processing.post_processing(predicts, 
                                                       scale_factors, 
                                                       img_ori_size, 
                                                       categories = self.artifacts.categories)
        return predicts
        
        


