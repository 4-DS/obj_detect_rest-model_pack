# Step CV-Pipeline: model_pack

This CV-Pipeline component is designed to convert the model into various formats (Onnx, OpenVino, TensorRT, etc.) and package the model in BentoService.   
When using the Binary Service, only the weights and the necessary files - configs for launching the service - are packaged.    
When using the REST API, in addition to packaging in the artifact bentoservice, the rest method predict, test_data, test_result is described   

Input data for step CV-Pipeline: model_pack
- **obj_detect_inference_files**     
Saved weights of the trained model (weights of the last epoch and with the best achieved metrics), configuration files from the previous CV-Pipeline step (model_train)

The output of this step CV-Pipeline is
- **bento_service**     
bento_service, packaged model service via BentoML (saved as a zip archive)

## How to run a step CV-Pipeline: model_pack

### Create a directory for the project (or use an existing one)
```
mkdir obj_detect_binary
cd obj_detect_binary
```  

### clone the repository: model_pack
```
git clone --recurse-submodules https://github.com/4-DS/obj_detect_binary-model_pack.git {dir_for_model_pack}
cd {dir_for_model_pack}
```  

### run step CV-Pipeline:model_pack
```
python step.dev.py
```  
or
```
step.prod.py
``` 
