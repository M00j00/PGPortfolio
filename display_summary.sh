get_id_trained(){
awk -F ',' '{print $1}' ./train_package/train_summary.csv | grep -v "net_dir"
}

models=$(get_id_trained)
for model in $models
do
	model_list=${model_list:+$model_list},$model
done
models=${model_list#,}
python main.py --mode=table --algos=$models --labels=$models
