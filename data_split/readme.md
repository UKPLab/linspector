This package contains the necessary files to create the probing tests. You need to unzip the `unimorph.zip` file under `../unzipped` directory, and download the word frequency lists for the languages you are interested in. We use [fastText vector files trained on wiki](https://fasttext.cc/docs/en/pretrained-vectors.html), since the words are ordered with frequency. Save the vector files under `../embeddings`.  

We already created the files for you under `../probing_datasets`. In case you want to generate with different settings or for different languages:

**Available Functions:**

* `prepareSingleTests.py` - Prepares single feature tests described in the paper
    * `args.feat`: If `1`, prepares unimorph related, single probing tasks e.g. Case, Gender, Tense
    * `args.common`: If `1`, creates the common tests POS, CharacterBin and TagCount  
    * `args.pseudo`: If `1`, prepares Wuggy tests. Reads the pseudowords from the `generated_wuggy_files` folder.
    * `args.nonlabelratio=0.3`: specifies the ratio of tokens with the 'None' label.   
    * `args.savedir`: The folder path of the probing tasks

* `preparePairedTests.py` - Prepares _OddFeat_ and _SameFeat_ tests     
    * `args.keepratio=0.2`: The desired rare word ratio  
    * `args.savedir`: The folder path of the probing tasks
