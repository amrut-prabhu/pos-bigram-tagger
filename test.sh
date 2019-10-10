#!/bin/bash
# File to store the accuracies of the POS tagger
TEST_RESULTS=test-results.txt
if [ ! -f $TEST_RESULTS ]
then
    touch $TEST_RESULTS
fi

COMMIT_HASH=$(git rev-parse HEAD)
echo "======$COMMIT_HASH======" >> $TEST_RESULTS

rm -f model-file sents.out
python buildtagger.py sents.train model-file
python runtagger.py sents.test model-file sents.out
ACCURACY_SENTS=$(python eval.py sents.out sents.answer)
echo $ACCURACY_SENTS
echo "sents $ACCURACY_SENTS" >> $TEST_RESULTS

echo

rm -f model-file-en en.out
python buildtagger.py en.train model-file-en
python runtagger.py en.test model-file-en en.out
ACCURACY_EN=$(python eval.py en.out en.answer)
echo $ACCURACY_EN
echo "en $ACCURACY_EN" >> $TEST_RESULTS
