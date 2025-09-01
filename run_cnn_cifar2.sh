#!/bin/bash

# Define arrays of sigma values, batch sizes, and seeds
sigmas=(3.0)  #2.0 4.0 6.0 8.0 10.0
batch_sizes=(1024) # 128 256
seeds=(42 44) # 123 456
red_rates=(0.3)

epochs=200

# --checkpoint-file "${checkpoint_file}" \
# --log-dir "${log_dir}" \
# --local_rank -1 \
# --device gpu \

for sigma in "${sigmas[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for seed in "${seeds[@]}"
        do

            for red_rate in "${red_rates[@]}"
            do
                # checkpoint_file="${checkpoint_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"
                # log_dir="${log_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"

                echo "---------------------------------------------------------"
                echo "  Sigma: $sigma, Batch size: $batch_size, Seed: $seed", 
                echo "---------------------------------------------------------"

                python fmnist.py \
                    --epochs ${epochs} \
                    --sigma ${sigma} \
                    --batch-size ${batch_size} \
                    --seed ${seed}\
                    --red_rate ${red_rate}
            
            done
            # sleep 10s
        done
    done
done
