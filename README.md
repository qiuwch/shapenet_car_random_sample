# shapenet_car_random_sample
Using shapenet cars to predict one target factor by random sample.

## How to generate CAM reuslts and calculate overlap score
### Generate part segmentation masks
First: modify settings in seg_dict_save.py into what you want
part_name: target part
seg_dir: directory of segmentation GT images
save_dir: directory to save the generated mask dictionary, the suggested format is .npy

Second: run the code
cd shapenet_car/
python seg_dict_save.py

### Calculate overlap score
Description: This .py file can be used with or without mask dictionary. If you don't have generated mask dictionary, it can generate automatically, but make sure to set all the settings right.

First: modify the configuration part in cam.py into what you want.
generate_cam: whether you want to generate CAM images, if so, you should set cam_dir.
cal_overlap: whether you want to calculate overlap score.

part_name: target part.
model_name: name of your trained CNN.

num_classes: number of classes in your output.
feature_extract: Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params.

test_dir: directory of test set
seg_dir: directory of segmentation images. If you have generated mask dictionary, this can be ignored.
seg_dict_dir: directory of mask dictionary. This is neccessary no matter if you have generated mask dictionary. If you have had the dictionary, this should be its directory; if not, this will be the location where it is saved.

param_dir: directory where you save your trained model; it should be in parameter mode.
pred_dir: directory of the predictions of your model on test set.

cam_dir: directory where you want to save your CAM results.
over_save_dir: directory where you want to save the overlap results, suggested in .csv format.
focus_dir: directory where you want to save the "focus" filenames.
unfocus_dir: directory where you want to save the "unfocus" filenames.
none_dir: directory where you want to save the "None" filenames.

Second: run the code
cd shapenet_car/
python cam.py
