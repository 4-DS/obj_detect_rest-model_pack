# Step CV-Pipeline: model_pack

During the CV Pipeline Model_Pack stage, the following steps take place:
1. Model conversion     
   The model trained in the previous CV-Pipeline Model_Train stage is converted into a format suitable for specific scenarios. For example, if the REST CV-Pipeline scenario is chosen, the model may be converted into the ONNX format, which enables deploying the model as a REST service. In the case of the Binary CV-Pipeline scenario, the model can be passed in PyTorch or another format in which it was trained.
2. Packaging into bentoservice     
   After model conversion, the model weights and all necessary artifacts (e.g., test image, predictions on the test image) are packaged into bentoservice. Packaging into bentoservice allows creating a containerized application that can be easily deployed and used for inference (prediction) on new data.

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
