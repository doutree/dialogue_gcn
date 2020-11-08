import argparse

from tqdm import tqdm
import pickle

import dgcn

log = dgcn.utils.get_logger()


def split():
    dgcn.utils.set_seed(args.seed)

    """
    videoIDs[vid] = List of utterance IDs in this video in the order of occurance
    videoSpeakers[vid] = List of speaker turns. e.g. [M, M, F, M, F]. here M = Male, F = Female
    videoText[vid] = List of textual features for each utterance in video vid.
    videoAudio[vid] = List of audio features for each utterance in video vid.
    videoVisual[vid] = List of visual features for each utterance in video vid.
    videoLabels[vid] = List of label indices for each utterance in video vid.
    videoSentence[vid] = List of sentences for each utterance in video vid.
    trainVid = List of videos (videos IDs) in train set.
    testVid = List of videos (videos IDs) in test set.
    """
    video_ids, video_speakers, video_labels, video_text, \
        video_audio, video_visual, video_sentence, trainVids, \
        test_vids = pickle.load(open('data/iemocap/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.5)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(dgcn.Sample(vid, video_speakers[vid], video_labels[vid],
                                 video_text[vid], video_audio[vid], video_visual[vid],
                                 video_sentence[vid]))
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(dgcn.Sample(vid, video_speakers[vid], video_labels[vid],
                               video_text[vid], video_audio[vid], video_visual[vid],
                               video_sentence[vid]))
    for vid in tqdm(test_vids, desc="test"):
        test.append(dgcn.Sample(vid, video_speakers[vid], video_labels[vid],
                                video_text[vid], video_audio[vid], video_visual[vid],
                                video_sentence[vid]))

    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test


def main(args):
    train, dev, test = split()
    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))
    data = {"train": train, "dev": dev, "test": test}
    dgcn.utils.save_pkl(data, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["iemocap", "avec", "meld"],
                        help="Dataset name.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    args = parser.parse_args()

    main(args)
