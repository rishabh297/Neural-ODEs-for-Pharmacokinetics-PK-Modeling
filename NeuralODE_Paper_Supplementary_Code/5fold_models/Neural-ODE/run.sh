
python Desktop/NeuralODE_Paper_Supplementary_Code/5fold_models/Neural-ODE/process_data.py
for fold in 1 2 3 4 5; do
    for model in 1 2 3 4 5; do
        python Desktop/NeuralODE_Paper_Supplementary_Code/5fold_models/Neural-ODE/data_split.py --data data.csv --fold $fold --model $model

        CUDA_VISIBLE_DEVICES="" python Desktop/NeuralODE_Paper_Supplementary_Code/5fold_models/Neural-ODE/run_train.py --fold $fold --model $model --save Desktop/NeuralODE_Paper_Supplementary_Code/5fold_models/Neural-ODE/results/fold_$fold --lr 0.00005 --tol 1e-4 --epochs 30 --l2 0.1
        CUDA_VISIBLE_DEVICES="" python Desktop/NeuralODE_Paper_Supplementary_Code/5fold_models/Neural-ODE/run_predict.py --fold $fold --model $model --save Desktop/NeuralODE_Paper_Supplementary_Code/5fold_models/Neural-ODE/results/fold_$fold --tol 1e-4
    done
done
