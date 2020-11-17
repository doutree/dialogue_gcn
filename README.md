### Requirements

- Python 3
- PyTorch 1.0
- PyTorch Geometric 1.3
- Pandas 0.23
- Scikit-Learn 0.20
- TensorFlow (optional; required for tensorboard)
- tensorboardX (optional; required for tensorboard)

### Execution

IEMOCAP dataset: `python train_IEMOCAP.py --base-model 'LSTM' --graph-model --nodal-attention --dropout 0.4 --lr 0.0003 --batch-size 32 --class-weight --l2 0.0 --no-cuda`