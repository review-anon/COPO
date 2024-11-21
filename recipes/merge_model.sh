source ~/.bashrc
bash ~/proxy.sh
module load curl 
module load anaconda/2022.10
module load cuda/12.1
source activate rlhf

export HF_ENDPOINT="https://hf-mirror.com"

echo "START TIME: $(date)"

export HUGGINGFACE_API_KEY="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"
export WANDB_API_KEY="a0b92158d55e0ca16cd94cbdebaef9117c99a118"

cd /home/baichenjia/COPO

i=2                      # TODO 注意
iter_num=3
username="baichenjia"
name="COPO-Llama-3-8B-Instruct"
fraction=$((61135/(iter_num)))
training_dataset="HuggingFaceH4/ultrafeedback_binarized"
model_name_or_path="data/${name}-iter-$((i-1))/final"        # TODO: load from local path
output_dir="data/${name}-iter-$((i-1))"                      # TODO: merge 

dataset_mixer="{'updated':'$username/${name}-dataset_iter_$i','original':'$training_dataset'}"
dataset_splits=("train_prefs[$((fraction*(i-1))):$((fraction*i))]","test_prefs")

python scripts/merge_model.py recipes/llama3-copo/copo_config_qlora.yaml model_name_or_path=$model_name_or_path dataset_mixer=$dataset_mixer dataset_splits=$dataset_splits output_dir=$output_dir
