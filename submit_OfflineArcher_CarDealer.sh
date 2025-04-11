#!/bin/bash -l
# SLURM SUBMIT SCRIPT (remove SLURM lines if running locally)

#SBATCH --nodelist=node-gpu01
#SBATCH --gres=gpu:1    # Request 1 GPU

# Optional environment setup
# . init.sh

# Hyperparameters
actor_lr=1e-4
critic_lr=1e-5
critic_expectile=0.9
inv_temp=1.0
batch_size=32
accumulate_grad_batches=4

# Training Command
python main.py fit \
--data=CarDealer \
--data.batch_size=$batch_size \
--data.n_traj_eval=64 \
--model=OfflineArcher \
--model.optimize_critic=True \
--model.actor_lr=$actor_lr \
--model.critic_lr=$critic_lr \
--model.discount_factor=0.99 \
--model.tau=0.05 \
--model.critic_expectile=$critic_expectile \
--model.inv_temp=$inv_temp \
--model.accumulate_grad_batches=$accumulate_grad_batches \
--trainer.fast_dev_run=False \
--trainer.max_epochs=10 \
--trainer.logger=WandbLogger \
--trainer.logger.init_args.project="CarDealer-Archer" \
--trainer.logger.init_args.name="AC_CarDealer_expectile_${critic_expectile}_invtemp_${inv_temp}" \
--trainer.strategy='ddp_find_unused_parameters_true' \
--trainer.val_check_interval=250
