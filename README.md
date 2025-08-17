# DARKINON
Hello! This is the official repository for our work DARKINON. Here, we aim to predict dark kinase-phosphosite associations.

In our previous study DARKIN, we provided a benchmark dataset and an evaluation of protein language models(pLMs). In this study, by using the best performing pLM, task-aware protein embeddings and developing a transformer based prediction model, we aimed to improve prediction performance compared to DARKIN and DeepKinZero (proposed by our lab).

### Dataset Format
| SUB_ACC_ID | SUB_MOD_RSD | SITE_+/-7_AA | KINASE_ACC_IDS |
|:---------|:---------|:---------|:---------|
|P68101|S52|MILLSELsRRRIRSI|Q9BQI3|
|P83268|S51|RILLsELsR______|Q9BQI3|
|P05198|S52|MILLsELsRRRIRsI|P28482,Q7KZI7,Q9BQI3,Q9NZJ5|
|P05198|S49|IEGMILLsELsRRRI|Q9BQI3,Q9NZJ5|

### Installation & Setting up the Environment

As the first step you have to download this repository to your local environment. You can either download it az a zip file or just clone the repository like this:

```
git clone git@github.com:tastanlab/darkinon.git
```

Now you have to create a conda environment to be able to run the code. Create the conda environment like this:

```
conda create --name darkin python=3.11.3
```

then activate this conda environment:

```
conda activate darkin
```

install pip if it is not installed:

```
conda install pip
```

now install the required packages. You could either use the requirements.txt file like this:

```
pip install -r requirements.txt
```

or you could directly install the required packges like this:

```
pip install pandas
pip install numpy
pip install matplotlib
```

now you should be all set to run the code!



### Running the Code


To run the transformer-based DARKINON Model, execute the `main.py`. This file supports several parameters:

- `mode`: Set this to either `'train'` or `'test'`.
- `config_path`: Specify the configuration `.yml` file you want to run.
- `num_of_models`: The number of models you wish to train.

**Example command:**
```bash
python -u main.py --mode train \
--config_path configs/to_carry/model_config.yaml \
--num_of_models 3
```

### Setting Up the YML Files

Example configuration files are located in the `configs/experiment_configs/transformer_finetuning` directory. While these files contain many parameters, you’ll typically only need to modify the following:

- `['phosphosite']['dataset']['train']`, `['validation']`, `['test']`:  
  Paths to your training, validation, and test datasets. These should point to the DARKIN dataset splits. You can use default paths like:  
  `../datasets/new_dataset/zsl/seed_12345/train_data.csv`.

- `['phosphosite']['dataset']['processor']['processor_type']`:  
  Specifies the protein language model (pLM) or word embedding used to encode phosphosites. Examples are available in the provided config files.

- `['phosphosite']['dataset']['processor']['model_name']`:  
  Specifies the protein language model (pLM) for phosphosites that will be fine-tuned during training. (We used our phosphosite-aware model.  huggingface : isikz/esm1b_mlm_pt_phosphosite)

- `['kinase']['dataset']['processor']['processor_type']`:  
  Specifies the protein language model or embedding method used for kinases. Again, examples can be found in the config files.

- `['kinase']['dataset']['processor']['kinase_embedding_path']`:  
  Path to the generated kinase embeddings. See tastanlab/darkin for embedding creation details. (We obtained kinase embeddings from our kinase-aware model. huggingface : isikz/esm1b_msa_mlm_ft_kinase)

- `['phosphosite']['model']['embedding_mode']` and `['kinase']['model']['embedding_mode']`:  
  Sets how the embeddings are used. Available options:
  - `cls`: Use the CLS token from the embedding.
  - `avg`: Average all token embeddings.
  - `sequence`: Use the full embedding sequence.
  - `concat`: Concatenate all token embeddings into one long vector.

- `['kinase']['dataset']['processor']`:  
  Contains optional toggles such as `use_family`, `use_group`, etc. Set these to `true` or `false` depending on your experimental preferences.

- `['hyper_parameters']`:  
  Contains model training hyperparameters (e.g., learning rate, batch size, etc.). Adjust these according to your experimental setup.
