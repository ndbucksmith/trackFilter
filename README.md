# KLF Training & Test

## Docker
### Building images
Variations on:
    $ docker build -f Dockerfile -t klf:tf-1.0.1-$(cat KLF_VERSION) .
    $ docker build -f Dockerfile.gpu -t klf:tf-1.0.1-gpu-$(cat KLF_VERSION) .

### Running Training
To just fire up training:
    $ nvidia-docker run -it --rm klf:tf-1.0.1-330e python train.py

To overlay a different KLF working dir, just mount it as a volume:
    $ nvidia-docker run -it --rm -v /home/corey/KLF:/KLF klf:tf-1.0.1-330e python train.py

Really, you're going to want environment goodness through:
    $ wdir=/tmp/$(cat KLF_VERSION)
    $ mkdir -p $wdir
    $ echo CUDA_VISIBLE_DEVICES=0 >> $wdir/train.env
    $ echo CHECKPOINT_BASEPATH=/test/checkpoint_%d >> $wdir/train.env
    $ nvidia-docker run -it --rm -v $wdir:/test --env-file=$wdir/train.env klf:tf-1.0.1-gpu-330e \
        python train.py >> $wdir/train.stdout 2>> $wdir/train.stderr
    # there really should be a wrapper script to write stdout/err to /test

### Running Test
    $ echo CUDA_VISIBLE_DEVICES= >> $wdir/test.env
    $ echo CHECKPOINT_PATH=/test/$(ls -1tr $wdir/checkpoint*index | tail -n1 | sed -e 's/.index$//' | xargs basename) >> $wdir/test.env
    $ nvidia-docker run -it --rm -v $wdir:/test --env-file=$wdir/test.env klf:tf-1.0.1-330e \
        python test.py >> $wdir/test.stdout 2>> $wdir/test.stderr

## Optional Environment Variables
| ENV VAR              | Train | Test | Notes |
|----------------------|:-----:|:----:|-------|
| CUDA_VISIBLE_DEVICES |  Opt  | Opt  | GPU device(s) for Tensorflow |
| RF_SERVER_ADDR       |  Opt  | Opt  | URI scheme + authority parts (e.g. http://192.168.1.229:8085) |
| CHECKPOINT_PATH      |  Opt  | Req  | Full path to checkpoint file to restore. |
| CHECKPOINT_BASEPATH  |  Opt  | n/a  | Full path to store checkpoints. May contain one %d placeholder which will be substituted with the training iteration number. |
| TRACE_BASEPATH       |  n/a  | Opt  | Full path to store debugging CSVs. |
| TRAINING_ITER        |  Opt  | n/a  | Number of iterations before quitting. |
| TRAINING_START_ITER  |  Opt  | n/a  | Optional, for resuming after a number of iterations. (e.g. when restoring trained model using CHECKPOINT_PATH) |
