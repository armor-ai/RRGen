# Automate Review Response Generation
Dataset and replication package for the Automating App Review Response Generation (ASE 2019).

## Usage
Run the code with
```angular2html
$ python model.py
```

You can change the configures in `parameter.py`, including the `hidden_size`, `word_vec_size`, `num_epochs`, etc. The important parameters are

```
use_sent_rate -- whether include sentence rating or not
use_sent_senti -- whether include sentence sentiment or not
use_sent_len -- define the sliced review length, i.e. the categorization interval, e.g., 20, else "False"
use_app_cate -- whether include app category or not
use_keyword -- whether include keyword information of one review or not
tie_ext_feature -- whether combine external features or not. "False" means that all the external features are not involved.
```

## Output
Some examples of generated reponses can be found in this [link](https://remine-lab.github.io/paper/rrgen.html).

## Dataset
As the dataset is very large and also such data can only be used for academic purpose, you need to fill [a requested form](https://docs.google.com/forms/d/1nTtDkpKrhiNjwkALjsQmhTvMFNf6H99Kn64NFWcIx9Y/edit?usp=sharing) first before downloading the data.

