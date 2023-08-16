## Dependency

- torch >=1.9.0
- transformers >=4.16.2

## Project Structure

Structure of the project

> |--Data
>
> ​	|--type2label_all.json (the mapping relationship between label id (numeric) and bugtype (string) )
>
> ​	|--type_map.md
>
> ​	|--train/val/test_data.csv (data used for experiment)
>
> |--Models
>
> ​	|--CompDefect ( the model proposed in the paper)
>
> ​	|--RQ1 (baselines, including CC2Vec, DeepJIT)
>
> ​	|--RQ2 (baselines, including FastText, TextCNN and pre-trained models like BERT, RoBERTa, CodeBERT)
>
> ​	|--RQ3 (baselines, including SequenceR, Codit, Edits, Recoder and NPR4j.zip we refer https://github.com/kwz219/NPR4J)
>
> ​	|--Disccusion (for Ablation Study)
>
> |--parser (tools used for extract data-flow，https://github.com/tree-sitter)
>
> |--utils
>
> ​	|-- myLogger.py
>
> ​	
>
> ​	

## CompDefect replication

```
cd Models/CompDefect
```

#### Training

To train the CompDefect model, you can use the command like this:

```
bash train_all.sh
```

#### Evaluating

To evaluate the model with test data, you can use the command like this:

```
bash eval_all.sh
```
