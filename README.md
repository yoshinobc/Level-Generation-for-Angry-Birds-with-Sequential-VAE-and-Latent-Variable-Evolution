# Level-Generation-for-Angry-Birds-with-Sequential-VAE-and-Latent-Variable-Evolution
This is an implementation of "Level Generation for Angry Birds with Sequential VAE and Latent Variable Evolution".

# Requirement
```bash
pip install -r requirements.txt
```
# Training
```train
python main.py
```
# Generating
When generating the level, specify the output directory after training as the --model_dir argument.
```generate
python main.py --generate --model_dir $TRAINED_DIR
```
Use
```
--model_epoch
```
to specify the epoch of the model during generation.
Use
```
--is_random
```
to generate it as a stochastic generator
# Loading
you can play them by loading them into ScienceBirds(https://github.com/lucasnfe/science-birds).

# Datasets
The dataset can be stored as an xml file in dataset/levels/, and can be created as a txt file by running make_levels.py.
The dataset that is already in there was created using [IratusAves](https://github.com/stepmat/IratusAves).
You will need to adjust some parameters such as the number of structures.
