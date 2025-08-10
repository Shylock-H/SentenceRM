# Usage
You need to preprocess the dataset to split it into sentences before training the sentence-level reward model.
```bash
python process.py
```
Then we recommend applying SFT (Supervised Fine-Tuning) to adapt the model to the chat template. It's observed that models trained with SFT perform better on RewardBench.
```bash
bash sh/train_sft.sh
```
Now you can train the sentence-level reward model by running the following command.
```bash
bash sh/train_srm.sh
```
After training, evaluate the reward model on RewardBench by running the following command.
```bash
bash sh/run_reward_bench.sh RM_PATH
```
