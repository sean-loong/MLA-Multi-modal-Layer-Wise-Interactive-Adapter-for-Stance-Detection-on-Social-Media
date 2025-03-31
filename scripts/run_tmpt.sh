read -p "input training dataset: [mtse, mccq, mwtwt, mruc, mtwq]: " trainDataset
read -p "input train dataset mode: [in_target, zero_shot]: " trainData
read -p "input model framework: [tmpt, tmpt_gpt_cot]: " framework
read -p "input model name: [bert_vit, roberta_vit, kebert_vit]: " trainModel
read -p "input running mode: [sweep, wandb, normal]: " runMode
read -p "input training cuda idx: " cudaIdx

# 输入训练集名字        ➜ trainDataset
# 选择训练模式          ➜ trainData
# 选择模型框架          ➜ framework
# 选择模型组合          ➜ trainModel
# 选择运行模式          ➜ runMode
# 选择 GPU 编号        ➜ cudaIdx

currTime=$(date +"%Y-%m-%d_%T")
fileName="run_tmpt.py"
outputDir="tmpt_logs/${trainData}"

if [ ! -d ${outputDir} ]; then
    mkdir -p ${outputDir}
fi

outputName="${outputDir}/${trainDataset}_${framework}_${trainModel}_${currTime}.log"
nohup python ${fileName} --cuda_idx ${cudaIdx} --dataset_name ${trainDataset} --model_name ${trainModel} --${trainData} --framework_name ${framework} --${runMode} > ${outputName} 2>&1 &

# --${someVar} 是一种 动态构建命令行参数名 的技巧，常用于控制多个布尔类型 flag 参数的开启，既简洁又高效，适合批量任务和脚本自动化。