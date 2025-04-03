# deep-bispectrum-inversion
DBI is a deep neural network for performing bispectrum inversion. The implementation builds upon convolutions and Swin transformers.
# Install
Install environmet using Anaconda
<pre> conda env create -f environment.yml </pre> 
Activate the environment
<pre> conda activate DBI_env </pre>
# Usage
# Training 
Example run:
<pre> python main.py --L 24 --K 2 --batch_size 100 --epochs 3000 --train_data_size 5000 --val_data_size 100 --data_mode random --scheduler OneCycleLR --optimizer AdamW --lr 4e-4 --loss_criterion mse --early_stopping --window_size 6 --num_heads 2 2 --depths 6 6 </pre>
Example for running with baseline data folder[^1]
<pre> python main.py --L 24 --K 2 --batch_size 100 --epochs 3000 --train_data_size 5000 --val_data_size 100 --data_mode random --scheduler OneCycleLR --optimizer AdamW --lr 4e-4 --loss_criterion mse --early_stopping --window_size 6 --num_heads 2 2 --depths 6 6 --read_baseline --baseline_data baseline_K_2_L_24_sz_100 </pre>

Configure parameters in config/params.py as needed.

# Training with DDP
Run with Distributed Data Parallel. Running with all available GPUs by default. 
<pre> torchrun main.py --L 24 --K 2 --batch_size 100 --epochs 3000 --train_data_size 5000 --val_data_size 100 --data_mode random --scheduler OneCycleLR --optimizer AdamW --lr 4e-4 --loss_criterion mse --early_stopping --window_size 6 --num_heads 2 2 --depths 6 6 </pre>

# Inference
After training, evalute the model visually and quantitatively. The output is saved into model_dir. 

<pre> python inference.py --model_dir <model_folder_path> --data_dir <data_folder_path> </pre>

Configure parameters in config/inference_params.py as needed (in inference, it also contains the cmd arguments).

[^1]: The baseline validation data is created using the [HeterogeneousMRA](https://github.com/NicolasBoumal/HeterogeneousMRA) repository.
