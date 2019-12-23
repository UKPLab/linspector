This package contains the necessary files to create the probing tests. 

# Generating non contextual probing tasks
You need to unzip the `unimorph.zip` file under `../unzipped` directory, and download the word frequency lists for the languages you are interested in. We use [fastText vector files trained on wiki](https://fasttext.cc/docs/en/pretrained-vectors.html), since the words are ordered with frequency. Save the vector files under `../embeddings`.  

We already created the files for you under `../intrinsic/data`. In case you want to generate with different settings or for different languages:

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
    
# Generate contextual probing tasks
You need to download the sigmorphon 2019 dataset from [here](https://github.com/sigmorphon/2019) and extract it to `../unzipped/sigmorphon19`]. Also you need the [Universal Dependencies treebank v2.4](http://hdl.handle.net/11234/1-2988). Please download it and extract it to `../unzipped/ud-treebanks-v2.4`.

    
* `prepareContextualTests.py` - Prepares all tests for the contextual probing tasks
    * `args.feat`: The tests which will be generated (separated by commas). All tests will be generated if this argument is not given.
    * `args.zscore`: Sentences which lengths do not match the z score will not be used for the probing tasks
    * `args.lang`: The languages (separated by commas) for which probing tasks will be generated. Tests for all languages will be generated if this argument is not given.
    * `args.size`: The number of instances for each probing task
    * `args.savedir`: The directory in which the probing tasks will be saved

    