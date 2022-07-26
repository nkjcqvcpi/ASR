# [A Challenging Benchmark of Anime Style Recognition](https://arxiv.org/abs/2204.14034v1)

---

_Update July 26: Preparing the ASR Dataset and releasing in the next few days._

---

This repository contains the Anime Style Recognition Dataset and the benchmark of anime style recognition tasks.

## Abstract
Given two images of different anime roles, anime style recognition (ASR) aims to learn abstract painting style to 
determine whether the two images are from the same work, which is an interesting but challenging problem. Unlike 
biometric recognition, such as face recognition, iris recognition, and person re-identification, ASR suffers from a much 
larger semantic gap but receives less attention. In this paper, we propose a challenging ASR benchmark. Firstly, we 
collect a large-scale ASR dataset (LSASRD), which contains 20,937 images of 190 anime works and each work at least has 
ten different roles. In addition to the large-scale, LSASRD contains a list of challenging factors, such as complex 
illuminations, various poses, theatrical colors and exaggerated compositions. Secondly, we design a cross-role protocol 
to evaluate ASR performance, in which query and gallery images must come from different roles to validate an ASR model 
is to learn abstract painting style rather than learn discriminative features of roles. Finally, we apply two powerful 
person re-identification methods, namely, AGW and TransReID, to construct the baseline performance on LSASRD. 
Surprisingly, the recent transformer model (i.e., TransReID) only acquires a 42.24% mAP on LSASRD. Therefore, we believe 
that the ASR task of a huge semantic gap deserves deep and long-term research.


[//]: # (## Licence)

[//]: # ()
[//]: # (For licence details, see [LICENCE]&#40;LICENCE&#41;.)

## Citation

If you find ASR useful, please citing:

```
@InProceedings{Li_2022_CVPR,
    author    = {Li, Haotang and Guo, Shengtao and Lyu, Kailin and Yang, Xiao and Chen, Tianchen and Zhu, Jianqing and Zeng, Huanqiang},
    title     = {A Challenging Benchmark of Anime Style Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {4721-4730}
}
```

[//]: # (## Acknowledgement)

[//]: # ()
[//]: # (This repository is built using the repository.)
