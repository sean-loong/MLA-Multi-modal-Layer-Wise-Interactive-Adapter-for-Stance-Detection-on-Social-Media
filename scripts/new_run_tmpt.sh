#!/bin/bash

# 设置默认值
defaultDataset="mtse"
defaultTrainData="in_target"
defaultFramework="adaptered_tmpt"
defaultModel="roberta_vit"
defaultRunMode="normal"
defaultcudaIdx="0"

# 带默认值的用户输入
read -p "Input training dataset [mtse, mccq, mwtwt, mruc, mtwq] (default: ${defaultDataset}): " trainDataset
trainDataset=${trainDataset:-$defaultDataset}

read -p "Input train dataset mode [in_target, zero_shot] (default: ${defaultTrainData}): " trainData
trainData=${trainData:-$defaultTrainData}

read -p "Input model framework [tmpt, tmpt_gpt_cot, adaptered_tmpt] (default: ${defaultFramework}): " framework
framework=${framework:-$defaultFramework}

read -p "Input model name [bert_vit, roberta_vit, kebert_vit] (default: ${defaultModel}): " trainModel
trainModel=${trainModel:-$defaultModel}

read -p "Input running mode [sweep, wandb, normal] (default: ${defaultRunMode}): " runMode
runMode=${runMode:-$defaultRunMode}

read -p "Input training CUDA index (e.g. 0) (default: ${defaultcudaIdx}): " cudaIdx
cudaIdx=${cudaIdx:-$defaultcudaIdx}


# 时间戳和路径设置
currTime=$(date +"%Y-%m-%d_%H-%M-%S")
fileName="new_run_tmpt.py"
outputDir="adaptered_tmpt_logs/${trainData}"

if [ ! -d ${outputDir} ]; then
    mkdir -p ${outputDir}
fi

outputName="${outputDir}/${trainDataset}_${framework}_${trainModel}_${currTime}.log"

# 构建运行命令
python ${fileName} \
    --cuda_idx ${cudaIdx} \
    --dataset_name ${trainDataset} \
    --model_name ${trainModel} \
    --${trainData} \
    --framework_name ${framework} \
    --${runMode} \
    --normal \
    # > ${outputName} 2>&1 &

echo "Training started with logs saved to ${outputName}"
