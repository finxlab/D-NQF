for data in brw mlo spo; do
    for learning_rate in 0.001 0.0001; do
        for batch_size in 32 64; do
            for moving_avg in 7 25 73; do
            	python run.py \
                       --data $data \
                       --learning_rate $learning_rate \
                       --batch_size $batch_size \
                       --moving_avg $moving_avg \
                       --train_epochs 100 \
                       --loss CRPS \
                       --n_quantiles 200 \
                       --model_type DLinear \
                       --task_name long_term_forecast
            done
        done
    done
done
