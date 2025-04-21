# deep-bispectrum-inversion
DBI is a deep neural network for performing bispectrum inversion. The implementation builds upon convolutions and Swin transformers.
# Install
Install environmet using Anaconda
<pre> conda env create -f environment.yml </pre> 
Activate the environment
<pre> conda activate DBI_env </pre>
# Usage
## Training 
Example run:
<pre> python main.py --L 24 --K 2 --batch_size 100 --epochs 3000 --train_data_size 5000 --val_data_size 100 --data_mode random --scheduler OneCycleLR --optimizer AdamW --lr 4e-4 --loss_criterion mse --early_stopping --window_size 6 --num_heads 2 2 --depths 6 6 </pre>
Example for running with data folder
<pre> python main.py --L 24 --K 2 --batch_size 100 --epochs 3000 --train_data_size 5000 --val_data_size 100 --data_mode random --scheduler OneCycleLR --optimizer AdamW --lr 4e-4 --loss_criterion mse --early_stopping --window_size 6 --num_heads 2 2 --depths 6 6 --read_baseline --baseline_data baseline_K_2_L_24_sz_100 </pre>

Configure parameters in config/params.py as needed.

## Training with DDP
Run with Distributed Data Parallel. Running with all available GPUs by default. 
<pre> torchrun main.py --L 24 --K 2 --batch_size 100 --epochs 3000 --train_data_size 5000 --val_data_size 100 --data_mode random --scheduler OneCycleLR --optimizer AdamW --lr 4e-4 --loss_criterion mse --early_stopping --window_size 6 --num_heads 2 2 --depths 6 6 </pre>

## Inference
After training, evalute the model visually and quantitatively. The output is saved into model_dir. 

<pre> python inference.py --model_dir <model_folder_path> --data_dir <data_folder_path> </pre>

Configure parameters in config/inference_params.py as needed (in inference, it also contains the cmd arguments).

## Acknowledgments

This work builds on the following open-source projects and publications:

- [**SwinIR**](https://github.com/JingyunLiang/SwinIR): This project incorporates the `network_swinir.py` file and the Residual Swin Transformer Block (RSTB) module.
  The original SwinIR repository is licensed under the MIT License, and the relevant components are reused and adapted here under the same license terms.  
  **Citation:**  
  Liang et al., *SwinIR: Image Restoration Using Swin Transformer*, ICCV Workshops, 2021.  
  [BibTeX](#bibtex-swinir)

- [**M-CNN**](https://github.com/tuan3w/m-cnn): Components such as the predefined `ConvBlock` and `ResnetBlock` were initially adapted from this repository, based on the MCNN paper.  
  **Citation:**  
  Arık et al., *Fast Spectrogram Inversion Using Multi-head Convolutional Neural Networks*, IEEE Signal Processing Letters, 2018.  
  [BibTeX](#bibtex-m-cnn)

- The validation data used in this project was generated using the bispectrum inversion method from [**HeterogeneousMRA**](https://github.com/NicolasBoumal/HeterogeneousMRA), which is also used as a baseline in our evaluation.  
  **Citation:**  
  Boumal et al., *Heterogeneous Multireference Alignment: A Single Pass Approach*, CISS 2018.  
  [BibTeX](#bibtex-heterogeneous-mra)

---

### BibTeX: SwinIR
```bibtex
@inproceedings{liang2021swinir,
  title={SwinIR: Image restoration using swin transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1833--1844},
  year={2021}
}
```

### BibTeX: M-CNN
```bibtex
@article{arik2018fast,
  title={Fast spectrogram inversion using multi-head convolutional neural networks},
  author={Arik, Sercan O and Jun, Heewoo and Diamos, Gregory},
  journal={IEEE Signal Processing Letters},
  volume={26},
  number={1},
  pages={94--98},
  year={2018}
}
```

### BibTeX: Heterogeneous MRA
```bibtex
@inproceedings{boumal2018heterogeneous,
  title={Heterogeneous multireference alignment: A single pass approach},
  author={Boumal, Nicolas and Bendory, Tamir and Lederman, Roy R and Singer, Amit},
  booktitle={2018 52nd Annual Conference on Information Sciences and Systems (CISS)},
  pages={1--6},
  year={2018},
  organization={IEEE}
}
```
