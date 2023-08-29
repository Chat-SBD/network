# Chat SBD - Network

This repository contains code to prepare training data and train our model, as well as the model itself.

## Structure

```
chopshop.py - Script to cut samples out of powerlifting videos, format them, and place them into training and testing folders.

train.py - Script to train the model on a batch, distributing computation across our Beowulf cluster.

batch/
    train/ - Holds training videos (90% of a batch).

    test/ - Holds testing videos (10%).

model/ - Holds the neural network model.
    analytics.txt - A file holding the performance stats of the new model (we might change this later).
```

## Training Pipeline

Per Batch:

1. Developer creates a new branch with the batch number and description of new data, if applicable (ex: '12_multiple-angles').
3. `chopshop.py` is run, possibly on a few different meet videos.
4. The resulting videos in `/test` and `/train` are checked for quality.
5. The videos are committed to the branch and the branch is published

---

5. The branch is pulled down onto the cluster master node.
6. `train.py` is run, which will train the model on the new training batch, test it on the testing batch, and report the results.
7. If nothing went wrong in the code, commit the new model and the statistics, and push the branch up.
8. Maybe we need to test the model on testing vids from every batch too? HEY GUYS READ THIS LINE THIS IS A TODO
9. If the new model is more accurate or improved in some way, merge the branch but DON'T DELETE IT! If for some reason it did not achieve the goal it was intended too, or maybe just confused the model, don't merge the branch - either just leave it or go back to the chop shop and try to improve the batch.
