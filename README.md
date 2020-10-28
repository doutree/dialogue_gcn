# dialogue_gcn
Pytorch implementation to paper "DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation". 

## Running
You can run the whole process very easily. Take the IEMOCAP corpus for example:

### Step 1: Preprocess.
```bash
./scripts/iemocap.sh preprocess
```

### Step 2: Train.
```bash
./scripts/iemocap.sh train
```

## Requirements
* Python 3
* PyTorch 1.0
* PyTorch Geometric 1.4.3
* Pandas 0.23
* Scikit-Learn 0.20
