#!/usr/bin/env bash

# EYEPACS EXPERIMENTS
#python train_lt_mxp.py --save_path eyepacs/mxp_1e-1/mobilenet --model_name mobilenetV2  --do_mixup  0.1 --n_epochs 30 --metric kappa
#python train_lt_mxp.py --save_path eyepacs/mxp_1e-2/mobilenet --model_name mobilenetV2  --do_mixup  0.2 --n_epochs 30 --metric kappa
#python train_lt_mxp.py --save_path eyepacs/mxp_1e-3/mobilenet --model_name mobilenetV2  --do_mixup  0.3 --n_epochs 30 --metric kappa
#python train_lt.py --save_path eyepacs/sqrt/mobilenet         --model_name mobilenetV2 --sampling sqrt --n_epochs 30 --metric kappa
#python train_lt.py --save_path eyepacs/class/mobilenet        --model_name mobilenetV2 --sampling class --n_epochs 30 --metric kappa
#python train_lt.py --save_path eyepacs/instance/mobilenet     --model_name mobilenetV2 --sampling instance --n_epochs 30 --metric kappa

# ENDOSCOPY EXPERIMENTS
python train_lt_mxp.py --save_path endo/F1/mxp_1e-1/mobilenetV2 --csv_train data/train_endo1.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F1/mxp_2e-1/mobilenetV2 --csv_train data/train_endo1.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F1/mxp_3e-1/mobilenetV2 --csv_train data/train_endo1.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F1/sqrt/mobilenetV2  --csv_train data/train_endo1.csv --data_path data/images        --model_name mobilenetV2 --sampling sqrt     --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F1/class/mobilenetV2  --csv_train data/train_endo1.csv --data_path data/images       --model_name mobilenetV2 --sampling class    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F1/instance/mobilenetV2  --csv_train data/train_endo1.csv --data_path data/images    --model_name mobilenetV2 --sampling instance --n_epochs 30 --metric mcc --n_classes 23

python train_lt_mxp.py --save_path endo/F2/mxp_1e-1/mobilenetV2 --csv_train data/train_endo2.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F2/mxp_2e-1/mobilenetV2 --csv_train data/train_endo2.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F2/mxp_3e-1/mobilenetV2 --csv_train data/train_endo2.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F2/sqrt/mobilenetV2  --csv_train data/train_endo2.csv --data_path data/images        --model_name mobilenetV2 --sampling sqrt     --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F2/class/mobilenetV2  --csv_train data/train_endo2.csv --data_path data/images       --model_name mobilenetV2 --sampling class    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F2/instance/mobilenetV2  --csv_train data/train_endo2.csv --data_path data/images    --model_name mobilenetV2 --sampling instance --n_epochs 30 --metric mcc --n_classes 23

python train_lt_mxp.py --save_path endo/F3/mxp_1e-1/mobilenetV2 --csv_train data/train_endo3.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F3/mxp_2e-1/mobilenetV2 --csv_train data/train_endo3.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F3/mxp_3e-1/mobilenetV2 --csv_train data/train_endo3.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F3/sqrt/mobilenetV2  --csv_train data/train_endo3.csv --data_path data/images        --model_name mobilenetV2 --sampling sqrt     --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F3/class/mobilenetV2  --csv_train data/train_endo3.csv --data_path data/images       --model_name mobilenetV2 --sampling class    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F3/instance/mobilenetV2  --csv_train data/train_endo3.csv --data_path data/images    --model_name mobilenetV2 --sampling instance --n_epochs 30 --metric mcc --n_classes 23


python train_lt_mxp.py --save_path endo/F4/mxp_1e-1/mobilenetV2 --csv_train data/train_endo4.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F4/mxp_2e-1/mobilenetV2 --csv_train data/train_endo4.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F4/mxp_3e-1/mobilenetV2 --csv_train data/train_endo4.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F4/sqrt/mobilenetV2  --csv_train data/train_endo4.csv --data_path data/images        --model_name mobilenetV2 --sampling sqrt     --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F4/class/mobilenetV2  --csv_train data/train_endo4.csv --data_path data/images       --model_name mobilenetV2 --sampling class    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F4/instance/mobilenetV2  --csv_train data/train_endo4.csv --data_path data/images    --model_name mobilenetV2 --sampling instance --n_epochs 30 --metric mcc --n_classes 23

python train_lt_mxp.py --save_path endo/F5/mxp_1e-1/mobilenetV2 --csv_train data/train_endo5.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F5/mxp_2e-1/mobilenetV2 --csv_train data/train_endo5.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23
python train_lt_mxp.py --save_path endo/F5/mxp_3e-1/mobilenetV2 --csv_train data/train_endo5.csv --data_path data/images --model_name mobilenetV2  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F5/sqrt/mobilenetV2  --csv_train data/train_endo5.csv --data_path data/images        --model_name mobilenetV2 --sampling sqrt     --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F5/class/mobilenetV2  --csv_train data/train_endo5.csv --data_path data/images       --model_name mobilenetV2 --sampling class    --n_epochs 30 --metric mcc --n_classes 23
python train_lt.py --save_path endo/F5/instance/mobilenetV2  --csv_train data/train_endo5.csv --data_path data/images    --model_name mobilenetV2 --sampling instance --n_epochs 30 --metric mcc --n_classes 23
