# deep-bispectrum-inversion
DBI is a deep neural network for performing bispectrum inversion. The implementation builds upon convolutions and Swin transformers.
# Install
Install environmet using Anaconda
<pre> conda env create -f environment.yml </pre> 
Activate the environment
<pre> conda activate DBI_env </pre>
# Usage
For creating a new validation data
<pre> python --L 24 --K 2 --batch_size 100 --epochs 3000 --train_data_size 5000 --val_data_size 100 --data_mode random --scheduler OneCycleLR --optimizer AdamW --lr 4e-4 --loss_criterion mse --early_stopping --window_size 6 --num_heads 2 2 --depths 6 6 </pre>
For running with a pre-created validation data, with bispectrum estaimations fitted to the MRA problem
<pre> python --L 24 --K 2 --batch_size 100 --epochs 3000 --train_data_size 5000 --val_data_size 100 --data_mode random --scheduler OneCycleLR --optimizer AdamW --lr 4e-4 --loss_criterion mse --early_stopping --window_size 6 --num_heads 2 2 --depths 6 6 --read_baseline --baseline_data baseline_K_2_L_24_sz_100 </pre>


