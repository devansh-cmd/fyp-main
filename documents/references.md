# Key References for FYP Report

## Attention Mechanisms (Core to Phase 5)

### Coordinate Attention
- **Hou, Q., Zhou, D., & Feng, J. (2021).** "Coordinate Attention for Efficient Mobile Network Design." *CVPR 2021*.
- DOI: 10.1109/CVPR46437.2021.01350
- https://arxiv.org/abs/2103.02907
- *Cited for: CA module design, direction-aware spatial attention on spectrograms*

### Attention Gate
- **Oktay, O., et al. (2018).** "Attention U-Net: Learning Where to Look for the Pancreas." *MIDL 2018*.
- https://arxiv.org/abs/1804.03999
- *Cited for: Attention Gate module, medical imaging attention mechanism*

### Squeeze-and-Excitation Networks
- **Hu, J., Shen, L., & Sun, G. (2018).** "Squeeze-and-Excitation Networks." *CVPR 2018*.
- DOI: 10.1109/CVPR.2018.00745
- https://arxiv.org/abs/1709.01507
- *Cited for: SE block, channel attention baseline*

### CBAM
- **Woo, S., et al. (2018).** "CBAM: Convolutional Block Attention Module." *ECCV 2018*.
- DOI: 10.1007/978-3-030-01234-2_1
- https://arxiv.org/abs/1807.06521
- *Cited for: Combined channel + spatial attention*

### Self-Attention / Transformers
- **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS 2017*.
- https://arxiv.org/abs/1706.03762
- *Cited for: Self-attention mechanism foundation*

---

## CNN Backbones

### ResNet
- **He, K., et al. (2016).** "Deep Residual Learning for Image Recognition." *CVPR 2016*.
- DOI: 10.1109/CVPR.2016.90
- https://arxiv.org/abs/1512.03385
- *Cited for: ResNet-50 backbone*

### MobileNetV2
- **Sandler, M., et al. (2018).** "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR 2018*.
- DOI: 10.1109/CVPR.2018.00474
- https://arxiv.org/abs/1801.04381
- *Cited for: MobileNetV2 backbone, lightweight branch in HybridNet*

---

## Audio & Spectrogram Processing

### Mel Spectrograms
- **Stevens, S.S., Volkmann, J., & Newman, E.B. (1937).** "A Scale for the Measurement of the Psychological Magnitude of Pitch." *JASA*.
- *Cited for: Mel scale foundation*

### Log-Mel for Audio Classification
- **Hershey, S., et al. (2017).** "CNN Architectures for Large-Scale Audio Classification." *ICASSP 2017*.
- DOI: 10.1109/ICASSP.2017.7952132
- https://arxiv.org/abs/1609.09430
- *Cited for: Using CNNs on log-mel spectrograms*

---

## Datasets

### ESC-50
- **Piczak, K.J. (2015).** "ESC: Dataset for Environmental Sound Classification." *ACM Multimedia 2015*.
- DOI: 10.1145/2733373.2806390
- https://github.com/karolpiczak/ESC-50
- *Cited for: ESC-50 dataset*

### EmoDB
- **Burkhardt, F., et al. (2005).** "A Database of German Emotional Speech." *Interspeech 2005*.
- DOI: 10.21437/Interspeech.2005-446
- *Cited for: EmoDB dataset*

### PhysioNet Heart Sounds
- **Clifford, G.D., et al. (2016).** "Classification of Normal/Abnormal Heart Sound Recordings." *Computing in Cardiology 2016*.
- DOI: 10.22489/CinC.2016.179-154
- *Cited for: PhysioNet 2016 heart sound challenge dataset*

### DementiaBank Pitt Corpus
- **Becker, J.T., et al. (1994).** "The Natural History of Alzheimer's Disease." *Archives of Neurology*.
- **MacWhinney, B. (2000).** "The CHILDES Project: Tools for Analyzing Talk." *Lawrence Erlbaum*.
- *Cited for: DementiaBank Pitt dataset, Cookie Theft description task*

### PC-GITA (Colombian Parkinson's)
- **Orozco-Arroyave, J.R., et al. (2014).** "New Spanish Speech Corpus and Measures of Dysarthria for the Automatic Detection of Parkinson's Disease." *Journal of the Acoustical Society of America*.
- DOI: 10.1121/1.4893891
- *Cited for: PC-GITA dataset, DDK task*

### Italian PD
- **Dimauro, G., et al. (2017).** "Assessment of Speech Intelligibility in Parkinson's Disease Using a Speech-To-Text System." *IEEE Access*.
- *Cited for: Italian Parkinson's Disease dataset*

---

## Transfer Learning & Training Methodology

### Transfer Learning for Audio
- **Kong, Q., et al. (2020).** "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." *IEEE/ACM TASLP*.
- DOI: 10.1109/TASLP.2020.3030497
- https://arxiv.org/abs/1912.10211
- *Cited for: Transfer learning from ImageNet to audio*

### K-Fold Cross-Validation
- **Kohavi, R. (1995).** "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection." *IJCAI 1995*.
- *Cited for: K-fold CV methodology*

### Stratified Group K-Fold
- **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine Learning in Python." *JMLR*.
- https://jmlr.org/papers/v12/pedregosa11a.html
- *Cited for: StratifiedGroupKFold, cross-validation implementation*

---

## Parkinson's Disease Detection from Speech

- **Sakar, C.O., et al. (2019).** "A Comparative Analysis of Speech Signal Processing Algorithms for Parkinson's Disease Classification." *Computer Speech & Language*.
- DOI: 10.1016/j.csl.2018.08.002
- *Cited for: PD speech classification survey*

- **Vasquez-Correa, J.C., et al. (2018).** "Multimodal Assessment of Parkinson's Disease: A Deep Learning Approach." *IEEE JBHI*.
- DOI: 10.1109/JBHI.2018.2866873
- *Cited for: Deep learning for PD from speech, DDK analysis*

---

## Hybrid / Ensemble Models

- **Gao, Z., et al. (2020).** "Hybrid Attention-based Deep Neural Network for Speech Enhancement." *IEEE/ACM TASLP*.
- *Cited for: Hybrid attention architectures in audio*

## Medical Image Attention

- **Schlemper, J., et al. (2019).** "Attention Gated Networks: Learning to Leverage Salient Regions in Medical Images." *Medical Image Analysis*.
- DOI: 10.1016/j.media.2019.01.012
- https://arxiv.org/abs/1808.08114
- *Cited for: Attention gates in medical imaging, directly inspired our AG module*
