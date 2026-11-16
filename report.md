---
title: Homework 1 -- Your Title Here
author: John Doe
course: Artificial Intelligence · UCU
date: 2026-10-12
---



## Results

| Model    | Accuracy | F1   |
|----------|----------|------|
| Baseline | 0.71     | 0.68 |
| Ours     | 0.84     | 0.82 |

Images go into the `images/` folder:

<!-- ![Training loss](images/loss.png) -->

## Conclusion

> Summarize your findings here.


## One Stage Anchor Free

For this model I chose YOLO26s - the latest YOLO model. It was a conscious decision to pick it instead of an older YOLOv8. I ran multiple experiments, none of which turned out very successful.

I was really betting on this model at the start, so I added a QUICK_TEST option to my notebook config for quick experiments. If it's true, the stride is increased to 15 and the model is only trained on first 10% of the dataset (this was later removed since it decreased the valuable variety of training data. I increased stride instead). This let me do quick experiments with 10 epochs running in ~10 minutes instead of a few hours.

Below you can see my experiments with this setting on:

### Test 1 (1024x1024 px)

![alt text](images/yolo/test1.png)

### Test 2 (640x640 px)

![alt text](images/yolo/test2.png)

Results were obviously worse so all future tests use 1024x1024 px.

### Test 3 (disabled NMS)

![alt text](images/yolo/test3.png)

YOLO26s predicts with its own one-to-one head, meaning it already predicts a single box for an object, so it doesn't need NMS.

### Test 4 (P2 head)

![alt text](images/yolo/test4.png)

Here I tried out enabling the forth P2 head of the YOLO model. Results look bad, but it's only because it probably takes longer to converge. A bit too long, in fact.

### Test 5 (same params as test 1 + ignore mask)

![alt text](images/yolo/test_mask.png)

I added an ignore mask which masks ignored regions on the training images, so the model doesn't get negative feedback for predicting objects that might still be there in the crowd

### Full run (test 4's config)

![alt text](images/yolo/check4_2.png)

Unfortunately, I didn't disable early stopping and the overnight run was finished before it could train for the full 20 epochs. I didn't try again since it wasn't possible to resume a run that's finished due to early stopping, and I wanted to compare the results of a no P2 model on a full proper run, since I was suspecting my QUICK_TEST runs weren't very reliable.

### Full run (test 1's config)

![alt text](images/yolo/check4_3_nop2.png)

The results of this run appeared really weird to me. But besides, you can clearly see the model was overfitting aggressively judging by diminishing train losses and rising val losses.

At this point I abandoned YOLO and continued my search for the endgame model.


## Query-based model

My first choice for the query-based model was RT-DETR-L. I trained it for ~6 hours, but it didn't produce any impressive results (~0.22 mAP on public leaderboard). Probably because the training image resolution was too low: only 640x640 px. I also saw no improvements after 15 epochs of training, so this first run taught me not to plan for this many epochs at the start. 

![alt text](images/detr.png)

The dip you can see in the Train losses graph at around 20th epoch is caused by Ultralytics image augmentation being turned off when less than 10 epochs of training are left, so the model quickly learns to predict the exact images it now sees in the training set. Before that, image augmentation was turned on, so the model didn't really see the exact images of the training set. Augmentation included flipping, color modification and patching, but was ultimately provided by Ultralytics contrary to a custom algorithm.

### Switching to D-FINE

I could tweak the parameters and probably get far better results out of this model, but after searching for "what's the best model for predicting the VisDrone dataset???" on Google I changed my mind and decided to train another query-based model instead - D-FINE. Their graphs on Github looked really impressive and blew RT-DETR out of the water.

![alt text](images/stats_padded.png)

My first run was on a COCO-pretrained D-FINE-M (20 mil params) with image resolution increased to 1024x1024. Right away - much better results: 0.3 on public leaderboard.

Then I got some very good news: it turned out that I had an RX 7800 XT I could use for local training with 16 GB of VRAM (instead of RTX 3070 8 GB I was using earlier), so this let me upgrade to D-FINE-L (30 mil params), and even increase my image resolution further to 896x1600 px. Other improvements I made on the next run are:

- Mask ignored regions on the training images, so the model doesn't get negative feedback for predicting objects that might still be there in the crowd
- Decrease learning rate (previous run overfitted quickly)
- In addition to (this time custom-implemented) generic image augmentation, I have downscaled 20% of the training images. Reasoning is that the test set contains a lot of low-res frames, while the training and validation sets almost have none, so this would help the model to learn to correctly detect blurred objects (because everything gets rescaled to 896x1600 during training and inference)

This gave even better results: 0.33 on the public leaderboard.

![alt text](images/dfine-l.png)

To make the submission, I used the model as it was at the 4th epoch, as it was deemed "best" in terms of mAP score for validation. You can see that it only decreases slightly afterwards on the graph. But I was still suspecting this could be false for the test set, especially since validation and test sets contain very different frame resolution distributions. So I tried predicting using the model at its last epoch state, and, sure enough, the score improved quite a lot (+0.12):

![alt text](images/photo_2026-11-16_20-28-03.jpg)

It's only now that I see the original submission had much higher private score (+0.1). I feel betrayed. Nevertheless, the submission_last.csv model ended up what we used for our final pipeline to build upon. Unfortunately.

## Our Pipeline

Basically what my "postprocessing" in the pipeline does is it divides every frame into 4 tiles, scales them up and predicts (using the loaded model) on them so small objects become bigger. Also, in addition to predicting on the test dataset, i also predict on its mirrored version, and then combine the predictions using WBF. And also temporal rescoring which simply raises the confidence score for a box in a frame if there are same boxes nearby in frames +-n and decreases it if there aren't. It's pretty basic. This is all I do in the pipeline, so all these changes are inference-time.

I decided not to build an ensemble since D-FINE was clearly way ahead of all of the other models we trained and they would only add noise. The only thing D-FINE still struggled the most with were small objects (mAP small was <0.16), but we weren't able to train a model that could improve on that specific task. Instead, to help with small objects, I use tiling, as I already explained.

### Results

During parameter search on validation set which can be seen in the notebook, it seemed that temporal rescoring was yielding minimal results no matter the window size or weight.

![alt text](<images/image copy.png>)

I still tried submitting with the best settings ("window": 2, "iou_thr": 0.5, "weight": 0.3) and it gave a subtly better result on public leaderboard, but on private it only harmed it. A lot. (0.295 without TR, 0.289 with)

![alt text](images/image.png)

Thankfully, we also tried implementing optical flow in the pipeline, and it gave promising results, helping us secure 5th place.

NO AI WAS USED IN THE MAKING OF THIS REPORT!!!!!!!!!!!!
