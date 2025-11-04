# TQx: Towards a text-based quantitative and explainable histopathology image analysis (MICCAI 2024 Early Accept)


## Overview

Implementation of the paper: 

> **TQx: Towards a text-based quantitative and explainable histopathology image analysis** \
> Anh Tien Nguyen, Trinh Thi Le Vuong and Jin Tae Kwak 

![Architecture](TQx.png)

### Abstract
> Recently, vision-language pre-trained models have emerged in computational pathology. Previous works generally focused on the alignment of image-text pairs via the contrastive pre-training paradigm. Such pre-trained models have been applied to pathology image classification in zero-shot learning or transfer learning fashion. Herein, we hypothesize that the pre-trained vision-language models can be utilized for quantitative histopathology image analysis through a simple image-to-text retrieval. To this end, we propose a Text-based Quantitative and Explainable histopathology image analysis, which we call TQx. Given a set of histopathology images, we adopt a pre-trained vision-language model to retrieve a word-of-interest pool. The retrieved words are then used to quantify the histopathology images and generate understandable feature embeddings due to the direct mapping to the text description. To evaluate the proposed method, the text-based embeddings of four histopathology image datasets are utilized to perform clustering and classification tasks. The results demonstrate that TQx is able to quantify and analyze histopathology images that are comparable to the prevalent visual models in computational pathology.

### Datasets
<ul>
  <li>Colon: <a href="https://github.com/QuIIL/KBSMC_colon_cancer_grading_dataset">link</a> </li>
  <li>WSSS4LUAD: <a href="https://wsss4luad.grand-challenge.org/WSSS4LUAD/">link</a></li>
  <li>Bladder: <a href="https://figshare.com/articles/dataset/Bladder_Whole_Slide_Dataset/8116043">link</a></li>
  <li>BACH: <a href="https://zenodo.org/records/3632035">link</a></li>
</ul>

### Model
QUILT-1M: https://huggingface.co/wisdomik/QuiltNet-B-16

### Word-of-interest (WoI)
All pathology terms are stored in `entity.csv` with 3 columns:
<ul>
  <li>id: Unified Medical Language System (UMLS) id of an entity (pathology word). </li>
  <li>entity_name: pathology word </li>
  <li>semantic_name: UMLS semantic type (category of pathology word) </li>
</ul>

### Image features
The text-based features of four datasets can be found in `result` folder. Each .pkl file stores all embeddings of a particular dataset with a specific setting.
For example: `all_img_features_sorted.pkl` store visual embeddings that are created from a visual encoder. `image_text_representation.pkl` in `Colon/Neoplastic_Process_1000/` stores text-based embeddings when the filter is `Neoplastic_Process`
Due to the size limitation, please find the Bladder results at [here](https://drive.google.com/drive/folders/1ay_uRoxTyzMBTg6bDHv5hFql1YPdY_62?usp=sharing).
