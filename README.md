# Concept-based Explanations for Out-Of-Distribution Detectors
This repository is the official implementation of the ICML 2023 paper: [Concept-based Explanations for Out-Of-Distribution Detectors](https://arxiv.org/pdf/2203.02586.pdf).

## Requirements
* It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requires some packages to be installed: see `environment.yml` for the list of dependencies of conda environment. 


## Running Experiments
* To learn concepts given a classifier (i.e., baseline): 
```
python concept_learn.py --name {NAME_OF_EXPERIMENT} --num_concept {NUM_CONCEPTS} --gpu {GPU}
                        --coeff_concept {COEFF_EXPL}
```
* To learn concepts given a classifier and a detector (i.e., ours):
```
python concept_learn.py --name {NAME_OF_EXPERIMENT} --num_concept {NUM_CONCEPTS} --gpu {GPU} 
                        --coeff_concept {COEFF_EXPL}
                        --ood 
                        --score {TYPE_OF_DETECTOR} --coeff_score {COEFF_MSE}
                        --feat_l2 --coeff_feat {COEFF_NORM} 
                        --separability --coeff_separa {COEFF_SEP}
```
* To evaluate detection completeness and concept separability:
```
python concept_eval.py --name {NAME_OF_EXPERIMENT} --result_dir {PATH_TO_CONCEPTS}
                       --out_data {OOD_DATASET} --gpu {GPU} 
                       --score {TYPE_OF_DETECTOR}
                       --separate
                       --visualize
```

## Acknowledgements
We build upon the baseline code by [Yeh et al., NeurIPS'20](https://github.com/chihkuanyeh/concept_exp).

## Citation
Please cite our work if you use this codebase:
```
@inproceedings{
choi2023concept-ood,
title={Concept-based Explanations for Out-of-Distribution Detectors},
author={Jihye Choi and Jayaram Raghuram and Ryan Feng and Jiefeng Chen and Somesh Jha and Atul Prakash},
booktitle={International Conference on Machine Learning},
year={2023}
}
```

## License
Please refer to the [LICENSE](https://github.com/jfc43/stratified-adv-rej/blob/main/LICENSE).
