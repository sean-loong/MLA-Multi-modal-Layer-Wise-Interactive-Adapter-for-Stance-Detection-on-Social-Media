# Multi-modal Layer-Wise Interactive Adapter for Stance Detection on Social Medi

This repository open-sources the code and datas in **Multi-modal Layer-Wise Interactive Adapter for Stance Detection on Social Media**.

Please cite our paper and kindly give a star for this repository if you use our code or data.

# Datasets

Noted: Due to Twitter's [developer agreement and privacy policy](https://developer.x.com/en/more/developer-terms/agreement-and-policy), researchers are only permitted to public Post IDs and User IDs. Therefore, in our GitHub repository, we open source the Post IDs and our annotation results. This approach aligns with the current mainstream practices in Twitter data research. The data is released for non-commercial research use.

## How to Hydrate

Please refer to [tools and libraries](https://developer.x.com/en/docs/twitter-api/tools-and-libraries/v2) to view hydrating tools supported by Twitter. [Tweepy](https://github.com/tweepy/tweepy) and [twarc](https://twarc-project.readthedocs.io/en/latest/twarc2_en_us/) are two commonly used tools. Before using these tools, you need to apply for the Twitter developer account. Afterwards, use these tools to supplement the content based on Post IDs.

##

For more details about the datasets, the please refer to the [data description](./dataset/Multi-Modal-Stance-Detection/README.md).

# Requirements

Seeing in requirement.txt

You could using `pip install -r requirement.txt` to install the required packages.

# Usage

Download your needed model weights into `model_state` or remove all `model_state/` dir prefix in all config files in `configs` to automatically download the model weights.

```
# baseline
sh scripts/run_baseline.sh

# MLA
sh scripts/run_MLA.sh
```

Take the CLIP model on in-target stance detection on mtse dataset for example:

```
>>> sh scripts/run_baseline.sh
>>> input training dataset: [mtse, mccq, mwtwt, mruc, mtwq]: mtse
>>> input train dataset mode: [in_target, zero_shot]: in_target
>>> input model framework: [textual, visual, multimodal]: multimodal
>>> input model name: [bert_vit, roberta_vit, kebert_vit, clip, vilt]: clip
>>> input running mode: [sweep, wandb, normal]: normal
>>> input training cuda idx: Your Cuda index
```

Our MLA on zero-shot stance detection on mwtwt dataset:

```
>>> sh scripts/run_MLA.sh
>>> input training dataset: [mtse, mccq, mwtwt, mruc, mtwq]: mwtwt
>>> input train dataset mode: [in_target, zero_shot]: zero_shot
>>> input model framework: [MLA, MLA_gpt_cot]: MLA
>>> input model name: [bert_vit, roberta_vit, kebert_vit]: bert_vit
>>> input running mode: [sweep, wandb, normal]: normal
>>> input training cuda idx: Your Cuda index
```

# Acknowledges
The code is implemented based on the excellent work: [Multi-Modal-Stance-Detection](https://github.com/Leon-Francis/Multi-Modal-Stance-Detection)
