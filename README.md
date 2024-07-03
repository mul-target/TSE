# TSE

paper: 

## Abstract



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
