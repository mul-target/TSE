# TSE

paper: A Two-Stage Multi-Target Stance Detection Approach Based on
Emotional Attention

## Abstract

Multi-target stance detection is a technique to detect authors' stances on multiple targets within the text. With the development of multi-target stance detection methods, attention has gradually shifted to the influence of emotions on stance, using emotional features to assist in stance detection. However, researchers have not explored the relationship between the two, merely using emotion tasks for assistance. In this paper, we propose a novel method, namely the two-stage multi-target stance detection method based on emotion, which enhances the auxiliary effect of emotion tasks by analyzing the relationship between stance and emotion. In the first stage, we obtain the stance labels of the targets through optimized knowledge distillation methods, then analyze the emotional labels of the text through prompt templates constructed by large language models, and compare the two types of labels after quantification. In the second stage, we analyze the data where the emotional and stance labels are inconsistent. We derive multiple topic words and emotion words embedded in the text through topic modeling and large language model techniques. We then obtain the final stance representation by fusing multiple features through shared attention. The experimental results show that our method achieves SOTA.

## Run

First, configure the environment:
```
$ pip install -r requirements.txt
```
For CAD in `merged` training setting, run
```
cd src/
python train_model.py \
    --input_target all \
    --model_select Bertweet \
    --train_mode unified \
    --col Stance1 \
    --lr 2e-5 \
    --batch_size 32 \
    --epochs 20 \
    --dropout 0. \
    --alpha 0.5
```
`input_target` can take one of the following target-pairs [`trump_hillary`, `trump_ted`, `hillary_bernie`] in adhoc setting and take [`all`] in merged setting.

`col` indicates the target in each target-pair. For example, for the target-pair `Trump-Clinton`, we have `Stance1` for Trump and `Stance2` for Clinton.

## Contact Info

Please contact Guantong Liu at lgt@dlmu.edu.cn with any questions.
