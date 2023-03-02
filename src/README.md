# The code

The code has been deliberately designed for re-usability. The class Pipeline (see pipeline.py) represents a re-usable 
module (somewhat similiar to sklearns pipeline), which accepts a pytorch dataset (with numpy elements) as inputs and 
outputs a new (transformed, predicted, etc.) dataset. Similarily, MultiPipelines are pipelines operating on multiple 
datasets at once.

To re-use part of the code, simply copy the relevent module over and pass it a pytorch dataset as input.