# SaTTCA
Implementation of "Scale-aware Test-time Click Adaptation for Pulmonary Nodule and Mass Segmentation"

Pulmonary nodules and masses are crucial imaging features in lung cancer screening that require careful management in clinical diagnosis. The segmentation performance on various sizes of lesions of nodule and mass is still challenging. Here, we use a multi-scale neural network with scale-aware test-time adaptation to address this challenge.<br>
<div align=center>
<img src="http://github.com//SplinterLi/SaTTCA/main/figures/introduction.png">
</div><br>
(a): Visualization on results of four large-scale mass segmentation given by nnUNet baseline.<br>
(b): Statistics of the number of nodules at different scales in three datasets.<br>
(c): The distribution of recall rate with respect to the nodule size.<br>
<div align=center>
<img src="http://github.com//SplinterLi/SaTTCA/main/figures/method.png">
</div>
The pipeline of the proposed Scale-aware Test-time Click Adaptation.

## Usage
Run train.py to train from scratch on your training set.<br>
Run sattc to do scare--aware test-time click adaptation on your test set.
