## Prepare data
The `./toolkit` folder contains scripts to prepare data.
### LINEMOD(LM6D_REFINE) and LINEMOD synthetic data(LM6D_REFINE_SYN)
Download the dataset from [http://ptak.felk.cvut.cz/6DB/public/datasets/hinterstoisser/](http://ptak.felk.cvut.cz/6DB/public/datasets/hinterstoisser/).
More specifically, only `test` have to be downloaded.
(Only the `test` folder contains real images which are used for training and testing in previous works, including ours)
Extract the `test` files to folder `$(DeepIM_root)/data/LINEMOD_6D/LM6d_origin`

Run these commands successively to prepare `LM6d_refine`:

Our processed models (`models.tar.gz`), train/val split (`LINEMOD_6D_image_set.tar.gz`) and PoseCNN's results (`PoseCNN_LINEMOD_6D_results.tar.gz`) can be found on [Google Drive](https://drive.google.com/drive/folders/1dxbEn9NOhlWjiEop3QPjT2wi-FB-N1if?usp=sharing)

Download and extract them in folder`$(DeepIM_root)/data/LINEMOD_6D/LM6d_converted/LM6d_refine`
which shall like:
```
LM6d_refine/models/ape, benchvise, ...
LM6d_refine/image_set/observed/ape_all.txt, ...
LM6d_refine/PoseCNN_LINEMOD_6D_results/ape, ...
```
After putting all the files in correct location, you can just run
```
sh prepare_data.sh
```
to prepare original dataset and synthetic data for LINEMOD.

## Prepare LINEMOD Occlusion data
Suppose you have prepared `LINEMOD_REFINE` in folder `$(DeepIM_root)/data/LINEMOD_6D/LM6d_converted/LM6d_refine`.

```
cd $(DeepIM_root)/data/LINEMOD_6D/LM6d_converted
ln -sf LM6d_refine/models models
mkdir -p LM6d_occ_render_v1/data
cd LM6d_occ_render_v1/data
ln -sf ../../LM6d_refine/data/observed
ln -sf ../../LM6d_refine/data/rendered
ln -sf ../../LM6d_refine/data/gt_observed
```

### LINEMOD Occlusion(LM6D_occ_v1)
Download `LINEMOD6D_OCC_image_set.tar.gz` from [Google Drive](https://drive.google.com/drive/folders/1dxbEn9NOhlWjiEop3QPjT2wi-FB-N1if?usp=sharing)
and extract the folder `image_set` into `LM6d_occ_render_v1`.

Download PoseCNN results on LINEMOD Occlusion from [Google Drive](https://drive.google.com/open?id=1AaOllZ-PS_5OyVv3WXimnL8stcWrd7WY),
extract the folder `results_occlusion_posecnn` into  `LM6d_occ_render_v1`.

In  `$(DeepIM_root)`, run
```
python toolkit/LM6d_occ_1_gen_train_pair_set.py
python toolkit/LM6d_occ_3_gen_PoseCNN_pred_rendered.py
```




### LINEMOD Occlusion synthetic data(LM6D_occ_dsm)
Download `LM6d_occ_dsm_train_observed_pose_all.pkl` from [Google Drive](https://drive.google.com/open?id=1xvh7apeZm1VXZLfpy0YWPc6anxLWOB0V)
and put it in `$(DeepIM_root)/data/LINEMOD_6D/LM6d_converted/LM6d_occ_dsm`.

In `$(DeepIM_root)`, run
```
python toolkit/LM6d_occ_dsm_1_gen_observed_light.py
python toolkit/LM6d_occ_dsm_2_gen_gt_observed.py
python toolkit/LM6d_occ_dsm_3_remove_low_visible.py
python toolkit/LM6d_occ_dsm_4_gen_rendered_pose.py
python toolkit/LM6d_occ_dsm_5_gen_rendered.py
```



We use indoor images from Pascal VOC 2012 ([download link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)) as the background of these synthetic during training.
Download and extract it in the `$(DeepIM root)/data`, which will like `$(DeepIM_root)/data/VOCdevkit/VOC2012`.

Support files for other dataset will be released later.
