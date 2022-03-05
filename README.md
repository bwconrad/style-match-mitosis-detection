# Style Match for Mitosis Figure Detection

## Setup
1. Run `pip install -r requirements.txt`.
2. Download the [MIDOG 2021 dataset](https://zenodo.org/record/4643381).
3. Optionally for STRAP, download the [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) and [COCO 2014 train](https://cocodataset.org/#download) datasets.

- Pretrained weights can be downloaded [here](https://drive.google.com/drive/folders/1t4e3AMhkeabucJ49arcuQSB80zOdmi0R?usp=sharing).

## Style Transfer Ablation Notebook
- `notebooks/style_transfer_inference.ipynb` provides a demo of the style transfer model qualitative ablation study.
- Model weights used in the notebook can be downloaded [here](https://drive.google.com/drive/folders/1t4e3AMhkeabucJ49arcuQSB80zOdmi0R?usp=sharing).

## Usage
### Style Transfer
- __Train__:
    - Content images from scanner 1 and style images from scanner 4: 
    ```
    python train_style_transfer_midog.py --gpus 1 --precision 16 --max_steps 80000 \
    --data.batch_size 16 --data.data_path data/midog --data.content_scanners 1 \
    --data.style_scanners 4 --model.use_bfg True --model.use_skip True \
    --check_val_every_n_epoch 100
    ```
    - Content and style images from all scanners:
    ```
    python train_style_transfer_midog.py --gpus 1 --precision 16 --max_steps 80000 \
    --data.batch_size 16 --data.data_path data/midog --data.content_scanners "[1,2,3,4]" \
    --data.style_scanners "[1,2,3,4]" --model.use_bfg True --model.use_skip True \
    --check_val_every_n_epoch 100
    ```
    - Content images from COCO and style images from WikiArt:
    ```
    python train_style_transfer.py --gpus 1 --precision 16 --max_steps 80000 \
    --data.batch_size 16  --data.resize_size 512  --data.content_path data/coco/ \
    --data.style_path data/wikiart/ --model.use_skip True --model.use_bfg True \
    --val_check_interval 5000
    ```
- __Evaluation__:
    - SSIM for model trained with content images from scanner 1 and style images from scanner 4:
    ```
    python test_style_transfer_midog.py --gpus 1 --precision 16 \
    --data.content_scanners 1 --data.style_scanners 4 \
    --checkpoint weights/adain_bfg_skip_c1_s4.ckpt
    ```

### Classification
- __Train__:
    - Scanner classification:
    ```
    python train_classifier.py --gpus 1 --precision 16 --max_epochs 5 \
    --data.data_path data/midog
    ```
- __Evaluation__:
    - On validation set:
    ```
    python test_classifier.py --gpus 1 --precision 16 --data.data_path data/midog \
    --checkpoint weights/classifier.ckpt
    ```
    - On validation set augmented by style transfer model:
    ```
    python test_classifier.py --gpus 1 --precision 16 --data.data_path data/midog \
    --data.style_scanner 4 --model.style_checkpoint weights/adain_bfg_skip_c1_s4.ckpt \
    --checkpoint weights/classifier.ckpt
    ```

### Detection
- __Train__:
    - Standard: 
    ```
    python train_detector.py --gpus 1 --precision 16 --max_epochs 100 --model.schedule step \
    --model.steps "[50]" --data.data_path data/midog --data.ann_path data/MIDOG.json \
    --data.train_scanners 1 --data.val_scanners 1
    ```
    - Style Match: 
    ```
    python train_detector.py --gpus 1 --precision 16 --max_epochs 100 --model.schedule step \
    --model.steps "[50]" --data.data_path data/midog --data.ann_path data/MIDOG.json \
    --data.train_scanners 1 --data.val_scanners 1 --data.style_scanner 2 \
    --model.style_checkpoint weights/all_scanners.ckpt
    ```
    - FDA: 
    ```
    python train_detector.py --gpus 1 --precision 16 --max_epochs 100 --model.schedule step \
    --model.steps "[50]" --data.data_path data/midog --data.ann_path data/MIDOG.json \
    --data.train_scanners 1 --data.val_scanners 1 --data.style_scanner 2 --data.fda_beta 0.01
    ```
    - STRAP: 
    ```
    python train_detector.py --gpus 1 --precision 16 --max_epochs 100 --model.schedule step \
    --model.steps "[50]" --data.data_path data/midog --data.ann_path data/MIDOG.json \
    --data.random_style_path data/wikiart/ --data.train_scanners 1 --data.val_scanners 1 \
    --model.style_checkpoint weights/random_style.ckpt
    ```
    - Stain Normalization: 
        - Apply stain normalization to the MIDOG dataset set by running `python scripts/stain_normalization -i data/midog -o data/normalizaed`.
    ```
    python train_detector.py --gpus 1 --precision 16 --max_epochs 100 --model.schedule step \
    --model.steps "[50]" --data.data_path data/normalized --data.ann_path data/MIDOG.json \
    --data.train_scanners 1 --data.val_scanners 1
    ```
- __Evaluation__:
    - Standard: 
    ```
    python test_detector.py --gpus 1 --precision 16 --data.data_path data/midog/ \
    --data.ann_path data/MIDOG.json --checkpoint weights/reg_s1.ckpt --data.test_scanners 3 \
    --model.eval_only_positives true
    ```
    - Style Match: 
    ```
    python test_detector.py --gpus 1 --precision 16 --data.data_path data/midog/ \
    --data.ann_path data/MIDOG.json --checkpoint weights/st_s1.ckpt --data.test_scanners 3 \
    --model.eval_only_positives true --data.style_scanners 2 \
    --model.style_checkpoint weights/all_scanners.ckpt
    ```
    - FDA: 
    ```
    python test_detector.py --gpus 1 --precision 16 --data.data_path data/midog/ \
    --data.ann_path data/MIDOG.json --checkpoint weights/fda_s1.ckpt --data.test_scanners 3 \
    --model.eval_only_positives true --data.style_scanners 2 --data.fda_beta 0.01 \
    --data.workers 0
    ```
    - STRAP: 
    ```
    python test_detector.py --gpus 1 --precision 16 --data.data_path data/midog/ \
    --data.ann_path data/MIDOG.json --data.random_style_path data/wikiart/ \
    --checkpoint weights/rand_s1.ckpt --data.test_scanners 3  --model.eval_only_positives true \
    --model.style_checkpoint weights/random_style.ckpt
    ```
    - Stain Normalization: 
    ```
    python test_detector.py --gpus 1 --precision 16 --data.data_path data/normalized/ \
    --data.ann_path data/MIDOG.json --checkpoint weights/norm_s1.ckpt --data.test_scanners 3 \
    --model.eval_only_positives true --data.style_scanners 2 \
    --model.style_checkpoint weights/all_scanners.ckpt
    ```
