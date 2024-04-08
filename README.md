# STAT 165 Final Project (SP 24)

### Environment Setup
Insert package installation commands here

### Data
1. Use data_prep folder for any scripts used to preprocess and prepare data for SFT / DPO
2. Use data folder to store raw and preprocessed datasets
3. Using the Story dataset and dataloaders as examples, create dataloaders for the forecasting datasets
4. Create an RLHF dataloader similar to the StoryRLHF dataloader but for forecasting data
5. Create a test set for evaluations (should contain about 150-200 examples)

### Training
1. Verify that dataloaders work correctly
2. Run SFT + DPO training for strong open-source model (size should be around 7B)
3. Merge any adapters that were trained before inference

### Inference + Evaluation
1. Write inference script to generate chat responses from trained model
2. Write test script to evaluate model performance on test dataset (will likely use similar logic as inference script, but use greedy decoding)




