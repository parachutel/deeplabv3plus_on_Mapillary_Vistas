# Semantic Segmentation on the Mapillary Vistas Dataset using the [DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab) [4] model by Google TensorFlow
## This is a repository for Stanford CS231N course project (spring 2018)
**Contact: Sheng Li ([parachutel_](https://github.com/parachutel)), available via lisheng@stanford.edu.**

The Mapillary Vistas Dataset is available for academic use at [here](https://www.mapillary.com/dataset/vistas?pKey=kBLk1dWR1ZuFPspBE9fN_w&lat=20&lng=0&z=1.5) (by request).

To build the dataset, put images in `/datasets/mvd/mvd_raw/JPEGImages/`, put ground truth labels in `/datasets/mvd/mvd_raw/SegmentationClass/`, put dataset split filename lists (text files) in `/datasets/mvd/mvd_raw/ImageSets/Segmentation/`. `/datasets/mvd/mvd_raw/ImageSets/Segmentation/build_image_sets.py` can help you build the dataset split list files. You will need to update `_MVD_INFORMATION` in `/datasets/segmentation_dataset.py` after building your dataset.

To preprocess the dataset and generate tfrecord files for faster reading, please run `/datasets/convert_mvd.sh`.

The initial model checkpoints are available [in the TensorFlow DeepLab Model Zoo ](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). Please put the ones you wish to use in `/datasets/mvd/init_models/`.

To run train, evaluate and visualize prediction using the model, use the following commands by running `local_test_mvd.sh` (you may comment out the parts you do not wish to run):

Train:
```
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --num_clones=4 \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=16 \
  --base_learning_rate=0.0025 \
  --learning_rate_decay_step=500 \
  --weight_decay=0.000015 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --log_steps=1 \
  --save_summaries_secs=60 \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_cityscapes_train/model.ckpt" \
  --initialize_last_layer=false \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${MVD_DATASET}"
```
Default value of --dataset is modified inside train.py directly.
Batch size and train_crop_size depends on your device's available memory.
 
Evaluation model:
```
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=<MAX_HEIGHT_PLUS> \
  --eval_crop_size=<MAX_WIDTH_PLUS> \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${MVD_DATASET}" \
  --max_number_of_evaluations=1
```

Visaulize the prediction:
```
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=<MAX_HEIGHT_PLUS> \
  --vis_crop_size=<MAX_WIDTH_PLUS> \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${MVD_DATASET}" \
  --max_number_of_iterations=1
```
**Note**: <MAX_HEIGHT_PLUS> and <MAX_WIDTH_PLUS> depends on the maximum resolution 
of your dataset. The following should be satisfied: <MAX_HEIGHT_PLUS> = output_stride * k + 1.
The default value, 513, is set for PASCAL images whose largest image dimension is 512.
We pick k = 32, resulting in eval_crop_size = 16 * 32 + 1 = 513 > 512. Same for <MAX_WIDTH_PLUS>.


============================================================================
# Original Documentation by Google TensorFlow DeepLab Developers:
# DeepLab: Deep Labelling for Semantic Image Segmentation

DeepLab is a state-of-art deep learning model for semantic image segmentation,
where the goal is to assign semantic labels (e.g., person, dog, cat and so on)
to every pixel in the input image. Current implementation includes the following
features:

1.  DeepLabv1 [1]: We use *atrous convolution* to explicitly control the
    resolution at which feature responses are computed within Deep Convolutional
    Neural Networks.

2.  DeepLabv2 [2]: We use *atrous spatial pyramid pooling* (ASPP) to robustly
    segment objects at multiple scales with filters at multiple sampling rates
    and effective fields-of-views.

3.  DeepLabv3 [3]: We augment the ASPP module with *image-level feature* [5, 6]
    to capture longer range information. We also include *batch normalization*
    [7] parameters to facilitate the training. In particular, we applying atrous
    convolution to extract output features at different output strides during
    training and evaluation, which efficiently enables training BN at output
    stride = 16 and attains a high performance at output stride = 8 during
    evaluation.

4.  DeepLabv3+ [4]: We extend DeepLabv3 to include a simple yet effective
    decoder module to refine the segmentation results especially along object
    boundaries. Furthermore, in this encoder-decoder structure one can
    arbitrarily control the resolution of extracted encoder features by atrous
    convolution to trade-off precision and runtime.

If you find the code useful for your research, please consider citing our latest
works:

*   DeepLabv3+:

```
@article{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  journal={arXiv:1802.02611},
  year={2018}
}
```

*   MobileNetv2:

```
@inproceedings{mobilenetv22018,
  title={Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation},
  author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
  booktitle={CVPR},
  year={2018}
}
```

In the current implementation, we support adopting the following network
backbones:

1.  MobileNetv2 [8]: A fast network structure designed for mobile devices.

2.  Xception [9, 10]: A powerful network structure intended for server-side
    deployment.

This directory contains our TensorFlow [11] implementation. We provide codes
allowing users to train the model, evaluate results in terms of mIOU (mean
intersection-over-union), and visualize segmentation results. We use PASCAL VOC
2012 [12] and Cityscapes [13] semantic segmentation benchmarks as an example in
the code.

Some segmentation results on Flickr images:
<p align="center">
    <img src="g3doc/img/vis1.png" width=600></br>
    <img src="g3doc/img/vis2.png" width=600></br>
    <img src="g3doc/img/vis3.png" width=600></br>
</p>

## Contacts (Maintainers)

*   Liang-Chieh Chen, github: [aquariusjay](https://github.com/aquariusjay)
*   YuKun Zhu, github: [yknzhu](https://github.com/YknZhu)
*   George Papandreou, github: [gpapan](https://github.com/gpapan)

## Tables of Contents

Demo:

*   <a href='https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb'>Colab notebook for off-the-shelf inference.</a><br>

Running:

*   <a href='g3doc/installation.md'>Installation.</a><br>
*   <a href='g3doc/pascal.md'>Running DeepLab on PASCAL VOC 2012 semantic segmentation dataset.</a><br>
*   <a href='g3doc/cityscapes.md'>Running DeepLab on Cityscapes semantic segmentation dataset.</a><br>
*   <a href='g3doc/ade20k.md'>Running DeepLab on ADE20K semantic segmentation dataset.</a><br>

Models:

*   <a href='g3doc/model_zoo.md'>Checkpoints and frozen inference graphs.</a><br>

Misc:

*   Please check <a href='g3doc/faq.md'>FAQ</a> if you have some questions before reporting the issues.<br>

## Getting Help

To get help with issues you may encounter while using the DeepLab Tensorflow
implementation, create a new question on
[StackOverflow](https://stackoverflow.com/) with the tags "tensorflow" and
"deeplab".

Please report bugs (i.e., broken code, not usage questions) to the
tensorflow/models GitHub [issue
tracker](https://github.com/tensorflow/models/issues), prefixing the issue name
with "deeplab".

## References

1.  **Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs**<br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille (+ equal
    contribution). <br />
    [[link]](https://arxiv.org/abs/1412.7062). In ICLR, 2015.

2.  **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,**
    **Atrous Convolution, and Fully Connected CRFs** <br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille (+ equal
    contribution). <br />
    [[link]](http://arxiv.org/abs/1606.00915). TPAMI 2017.

3.  **Rethinking Atrous Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](http://arxiv.org/abs/1706.05587). arXiv: 1706.05587, 2017.

4.  **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. arXiv: 1802.02611.<br />
    [[link]](https://arxiv.org/abs/1802.02611). arXiv: 1802.02611, 2018.

5.  **ParseNet: Looking Wider to See Better**<br />
    Wei Liu, Andrew Rabinovich, Alexander C Berg<br />
    [[link]](https://arxiv.org/abs/1506.04579). arXiv:1506.04579, 2015.

6.  **Pyramid Scene Parsing Network**<br />
    Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia<br />
    [[link]](https://arxiv.org/abs/1612.01105). In CVPR, 2017.

7.  **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate shift**<br />
    Sergey Ioffe, Christian Szegedy <br />
    [[link]](https://arxiv.org/abs/1502.03167). In ICML, 2015.

8.  **Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation**<br />
    Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen<br />
    [[link]](https://arxiv.org/abs/1801.04381). arXiv:1801.04381, 2018.

9.  **Xception: Deep Learning with Depthwise Separable Convolutions**<br />
    François Chollet<br />
    [[link]](https://arxiv.org/abs/1610.02357). In CVPR, 2017.

10. **Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge 2017 Entry**<br />
    Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei, Jifeng Dai<br />
    [[link]](http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf). ICCV COCO Challenge
    Workshop, 2017.

11. **Tensorflow: Large-Scale Machine Learning on Heterogeneous Distributed Systems**<br />
    M. Abadi, A. Agarwal, et al. <br />
    [[link]](https://arxiv.org/abs/1603.04467). arXiv:1603.04467, 2016.

12. **The Pascal Visual Object Classes Challenge – A Retrospective,** <br />
    Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John
    Winn, and Andrew Zisserma. <br />
    [[link]](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). IJCV, 2014.

13. **The Cityscapes Dataset for Semantic Urban Scene Understanding**<br />
    Cordts, Marius, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele. <br />
    [[link]](https://www.cityscapes-dataset.com/). In CVPR, 2016.
