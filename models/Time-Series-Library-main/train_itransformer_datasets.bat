@echo off
cd /d %~dp0

echo Training iTransformer on Electricity...
python run.py --task_name long_term_forecast --is_training 1 --root_path ../../datasets/electricity/ --data_path electricity.csv --model_id ECL_96_96 --model iTransformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 512 --d_ff 2048 --batch_size 32 --train_epochs 10 --learning_rate 0.0005 --des Exp --itr 1 --freq h

echo Training iTransformer on Exchange Rate...
python run.py --task_name long_term_forecast --is_training 1 --root_path ../../datasets/exchange_rate/ --data_path exchange_rate.csv --model_id Exchange_96_96 --model iTransformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 512 --d_ff 2048 --batch_size 32 --train_epochs 10 --learning_rate 0.0005 --des Exp --itr 1 --freq d

echo Training iTransformer on Traffic...
python run.py --task_name long_term_forecast --is_training 1 --root_path ../../datasets/traffic/ --data_path traffic.csv --model_id Traffic_96_96 --model iTransformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 512 --d_ff 2048 --batch_size 32 --train_epochs 10 --learning_rate 0.0005 --des Exp --itr 1 --freq h

echo Training iTransformer on Weather...
python run.py --task_name long_term_forecast --is_training 1 --root_path ../../datasets/weather/ --data_path weather.csv --model_id Weather_96_96 --model iTransformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_model 512 --d_ff 2048 --batch_size 32 --train_epochs 10 --learning_rate 0.0005 --des Exp --itr 1 --freq t