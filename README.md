# Parallel Knowledge Enhancement based Framework for Multi-behavior Recommendation

This repository contains TensorFlow codes and datasets for the paper.

## Environment
The code has been tested running under Python 3.6.15. The required packages are as follows:
* nvidia-tensorflow == 1.15.4+nv20.10
* tensorflow-determinism == 0.3.0
* numpy == 1.19.5
* scipy == 1.7.3

## Datasets
We utilized three datasets to evaluate PKEF: <i>Beibei, Taobao, </i>and <i>Tmall Contest</i>. The <i>purchase</i> behavior is taken as the target behavior for all datasets. The last target behavior for the test users are left out to compose the testing set. We filtered out users and items with too few interactions.

## Just Run It！

* Beibei
```
python PKEF_final.py --data beibei
```
* Taobao
```
python PKEF_final.py --data taobao
```
* Tmall
```
python PKEF_final.py --data tmall --gnn_layer "[4, 1, 1, 1]" --coefficient "[0.0/6, 4.0/6, 0.0/6, 2.0/6]"
```

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@article{meng2023parallel,
  title={Parallel Knowledge Enhancement based Framework for Multi-behavior Recommendation},
  author={Meng, Chang and Zhai, Chenhao and Yang, Yu and Zhang, Hengyu and Li, Xiu},
  journal={arXiv preprint arXiv:2308.04807},
  year={2023}
}
```



