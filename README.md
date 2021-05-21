Repository for holding development of generalization of Active Inference.

Stages of development are as follows:

1. Extension of state space by building a generative model using VAE
2. Build a generative model (the State model) including a posterior model, a likelihood model and a state transition model. Also implemented  _planning as inference_ framework.  
Reference: [Çatal, Ozan, et al. "Learning generative state space models for active inference." Frontiers in Computational Neuroscience 14 (2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7701292/).
4. Applied to OpenAI gym classical control problems, e.g. MountainCar-v0.
5. Build an active inference agent which learns its local prior preference (the PPL model) from human expert data. Techniques learned from RL such as Prioritized Experience Replay are applied to improve the agent performance.  
Reference: [Shin, Jinyoung, Cheolhyeong Kim, and Hyung Ju Hwang. "Prior Preference Learning from Experts: Designing a Reward with Active Inference." arXiv preprint arXiv:2101.08937 (2021)](https://arxiv.org/abs/2101.08937).
6. Build an interception task environment based on OpenAI gym to work with AIF agent.
7. Build a 3D interception task environment with Unity3D for human data collection and ML experiments.

Stretch goals:
1. Compare our AIF agent with modern RL agents on benchmark tasks.
2. Replace backprop with predictive processing methods for biological plausibility and generalizability.

To run the code for the PPL model, you can use the following Bash commands:  

First, set hyperparameters in the config file, e.g. for MountainCar-v0, use <code>/configs/MountainCar-v0.config</code>.
To fit/train a local prior model to expert data (imitation learning), which will be later used in the PPL agent, run the following command:
<pre>
$ python train_prior_model.py -c configs/MountainCar-v0.config
</pre>
To train the final PPL agent, run the following command:
<pre>
$ python train_ppl_model.py -c configs/MountainCar-v0.config
</pre>
