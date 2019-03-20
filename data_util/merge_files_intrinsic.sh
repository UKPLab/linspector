#!/usr/bin/env bash


WORDFOLDER=../words/intrinsic_lower
EMBEDFOLDER=../saved_embeddings/intrinsic_lower

slang="de"

# merge words and embeddings with space delimiter
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/elmo/embeds.vec > $EMBEDFOLDER/$slang/elmo/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/fasttext/embeds.vec > $EMBEDFOLDER/$slang/fasttext/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/muse_supervised/embeds.vec > $EMBEDFOLDER/$slang/muse_supervised/final_embeds.vec


