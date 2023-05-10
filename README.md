## End To End ML Project

### created a environment
```
conda create -p envci python==3.8

conda activate envci/
```
### Install all necessary libraries
```
conda deactivate
pip install -r requirements.txt

conda info
conda deactivate
conda remove --name ENV_NAME --all
### Execution path
src/pipeline/training_pipeline.py
src/pipeline/prediction_pipeline.py

