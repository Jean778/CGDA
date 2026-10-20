# -*- coding: utf-8 -*-
"""Build model->reference mapping table for the two Table IX comparisons."""

from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repository root

import pandas as pd
from pathlib import Path

p = REPO / 'main_results'

rows = [
    # Table IX of OTTA-DriftNet (IEEE TIM 2025, Liang et al.)
    dict(table='OTTA-DriftNet Table IX', model='ODCNN', params_M=0.265766, flops_M=0.212848,
         reference='Xie et al., "Application of a new one-dimensional deep convolutional neural network for intelligent fault diagnosis of rolling bearings," Science Progress, 2020, DOI 10.1177/0036850420951394 (1-D CNN; not confirmed as the exact source used in the table)',
         status='uncertain'),
    dict(table='OTTA-DriftNet Table IX', model='VGG13', params_M=9.604422, flops_M=25.843610,
         reference='Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR 2015, arXiv:1409.1556',
         status='confirmed'),
    dict(table='OTTA-DriftNet Table IX', model='GasNet', params_M=0.382118, flops_M=1.222308,
         reference='Peng, Zhao, Pan, Ye, "Gas Classification Using Deep Convolutional Neural Networks," Sensors, 18(1):157, 2018, DOI 10.3390/s18010157 (38-layer DCNN named GasNet)',
         status='confirmed'),
    dict(table='OTTA-DriftNet Table IX', model='TDACNN', params_M=1.362544, flops_M=2.568145,
         reference='Zhang, "TDACNN: Target-domain-free domain adaptation convolutional neural network for drift compensation in gas sensors," Sens. Actuators B: Chem., 361:131739, 2022 (listed in OTTA-DriftNet ref [16])',
         status='confirmed'),
    dict(table='OTTA-DriftNet Table IX', model='M2FL-CCC', params_M=0.745586, flops_M=1.098821,
         reference='Zhai, Han, Li, Yang, Duan, Yan, "M2FL-CCC: Multibranch multilayer feature leaning and comprehensive classification criterion for gas sensor drift compensation," IEEE TIM, 72:1-12, 2023 (listed in OTTA-DriftNet ref [17])',
         status='confirmed'),
    dict(table='OTTA-DriftNet Table IX', model='SniffConv', params_M=0.183154, flops_M=6.055386,
         reference='Serra et al., "Validation of the rapid detection approach for enhancing the electronic nose systems performance, using different deep learning models and support vector machines," arXiv:2005.01611, 2020 (Sniff ConvNet, one of the SniffNets family)',
         status='confirmed'),
    dict(table='OTTA-DriftNet Table IX', model='SniffMultinose', params_M=0.219486, flops_M=6.682468,
         reference='Serra et al., arXiv:2005.01611, 2020 (Sniff Multinose, fusion model of the SniffNets family)',
         status='confirmed'),
    dict(table='OTTA-DriftNet Table IX', model='ODAELM', params_M=0.167239, flops_M=5.869713,
         reference='Ma, Luo, Qin, et al., "Online sensor drift compensation for E-nose systems using domain adaptation and extreme learning machine," Sensors, 18(3):742, 2018, DOI 10.3390/s18030742',
         status='confirmed'),
    dict(table='OTTA-DriftNet Table IX', model='OTTA-DriftNet (proposed)', params_M=0.446973, flops_M=5.137291,
         reference='Liang, Chen, Yang, Xie, Tian, Liu, IEEE TIM, 74, 2025, DOI 10.1109/TIM.2025.3591198',
         status='self'),

    # Table IX of MLAEC-Net (IEEE Sensors J. 2024, Liang et al.)
    dict(table='MLAEC-Net Table IX', model='LSTM', params_M=0.187862, flops_M=0.230656,
         reference='Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation, 9(8):1735-1780, 1997 (listed in MLAEC-Net ref [19])',
         status='confirmed'),
    dict(table='MLAEC-Net Table IX', model='VGG19', params_M=20.032690, flops_M=45.967070,
         reference='Simonyan & Zisserman, ICLR 2015, arXiv:1409.1556',
         status='confirmed'),
    dict(table='MLAEC-Net Table IX', model='LeNet', params_M=0.258446, flops_M=4.284246,
         reference='LeCun et al., "Gradient-based learning applied to document recognition," Proc. IEEE, 86(11):2278-2324, 1998',
         status='confirmed'),
    dict(table='MLAEC-Net Table IX', model='AlexNet', params_M=5.734214, flops_M=19.399150,
         reference='Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NeurIPS 2012',
         status='confirmed'),
    dict(table='MLAEC-Net Table IX', model='GoogLeNet', params_M=2.979702, flops_M=9.489656,
         reference='Szegedy et al., "Going Deeper with Convolutions," CVPR 2015',
         status='confirmed'),
    dict(table='MLAEC-Net Table IX', model='SniffMultinose', params_M=0.219486, flops_M=6.682468,
         reference='Serra et al., arXiv:2005.01611, 2020',
         status='confirmed'),
    dict(table='MLAEC-Net Table IX', model='ResNet34', params_M=23.850750, flops_M=28.636210,
         reference='He et al., "Deep Residual Learning for Image Recognition," CVPR 2016',
         status='confirmed'),
    dict(table='MLAEC-Net Table IX', model='ODCNN', params_M=0.265766, flops_M=0.212848,
         reference='Xie et al., Science Progress, 2020, DOI 10.1177/0036850420951394 (1-D CNN; not confirmed as the exact source used in the table)',
         status='uncertain'),
    dict(table='MLAEC-Net Table IX', model='GasNet', params_M=0.382118, flops_M=1.222308,
         reference='Peng et al., Sensors, 18(1):157, 2018, DOI 10.3390/s18010157',
         status='confirmed'),
    dict(table='MLAEC-Net Table IX', model='MLAEC-Net (proposed)', params_M=0.568672, flops_M=1.108704,
         reference='Liang, Chen, Yang, Chen, IEEE Sensors J., 24(16):25830-25841, 2024, DOI 10.1109/JSEN.2024.3418849',
         status='self'),
]

t = pd.DataFrame(rows)
t.to_csv(p / 'table_ix_model_references.csv', index=False)
pd.set_option('display.width', 240)
pd.set_option('display.max_colwidth', 70)
print(t.to_string(index=False))
print()
print('saved -> main_results/table_ix_model_references.csv')
