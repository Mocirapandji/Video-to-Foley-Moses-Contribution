# Video-to-Foley-Moses-Contribution
## A contribution of the code I did for Masterclass project

# 1. Mainly finished the Data-loader and Dataset

Within the project I mainly focused on creating the Dataset. This Dataset was complicated because we had to create our own dataset footsteps since there was no existing database about it.

Within the code, I have made 3 different version all improving upon each other.

## Version 1:
Automated machine-learning-powered extraction pipeline. I used YOLOv8 to physically track the person ankle and synchronize it to the movement audio spikes from librosa to find the footsteps itself. More of a proof of concept.

### Version 1 --> 2:
1. Cleaned the Directory
2. Dataset Restructuring
3. Added threshold tuning for quieter sounds
4. updated slice_and_save constraints
5. Code Cleanup


## Version 2:
Refined extraction pipeline optimized for subtle foley. I lowered the audio onset detection thresholds to capture softer, low-decibel impacts like walking on crunchy leaves and relaxed the slicing constraints to keep shorter, high-quality sequences. Focused on sensitivity tuning and cleaner dataset organization.

### Version 2 --> 3:
1. Youtube Authentication to bypass videos that are flagged as age-restricted
2. Completely changed the slicing logic to create more uniform length clips using the slidign window approach.
3. Automated the train and test splitting so that it can create clean folders for both

## Version 3(Final):
Production-ready dataset generator. I replaced the variable-length cuts with a sliding-window algorithm to create perfectly uniform 3-second overlapping clips, standardizing the data for the model. I also added cookie authentication to bypass YouTube bot-blockers and built an automated 80/20 train/test splitter to cleanly route the final media into the Dataset_train and Dataset_test directories.


# 2. Worked on the Machine learning

Within the project after the dataloader I directly also tried helping with multiple versions of the machine. The directory that says "Other_Training" is where all the other files that I helped trying to change small bits and use experimental things like HiFi-GAN and mixing transformers.

The main file I worked on and contributed to making a good video-to-foley sound is "FullTraining(Moses).py"
