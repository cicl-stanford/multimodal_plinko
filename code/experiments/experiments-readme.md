# Experiments readme

## Prediction

The prediction task was deployed using [psiturk](https://psiturk.org/) on python2. `psiturk_env.yml` contains the package requirements and can be used to setup a working python environment.

Once psiturk is installed, the experiment can be run in "debug" mode. See [psiturk documentation](https://psiturk.org/) for details.

## Inference

The inference task was developed using [psychopy](https://www.psychopy.org/) originally on python2. The original package environment was not recorded, and unfortunately we could not reconstruct the experiment in its original version. The code here is an updated version of the experiment with minimal modifications made to adapt the experiment to python3. `psychopy_env.yml` contains the package requirements and can be used to setup a working python environment.

The three subfolders in inference correspond to the conditions of the inference task as follows:

- no sound + ball visible --> vision
- sound + ball visible --> audio
- sound + ball occluded --> occluded

We use this correspondence throughout the repo to refer to the different inference conditions.

To run the code for a particular condition, navigate to the `exp_code` subfolder within the condition directory and run the following:

```
python plinko_eytracking.py
```

By default, the experiment code is set to testing mode with a dummy eyetracker, and demographics collection is turned off. These settings can be modified with the global variables at the top of the `plinko_eytracking.py` script.