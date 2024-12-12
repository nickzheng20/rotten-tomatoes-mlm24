import torch
import os
import pandas as pd
import argparse
from utils.utils import encode_mask, load_and_decode, display_image_and_mask
from utils.data_loading import TomatoLeafDataset
from torch.utils.data import DataLoader
from unet.model import *
from matplotlib import pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Predict the tomato leaf mask")
    parser.add_argument('-i', '--image', action='store_true', help="Generate and save prediction images")
    parser.add_argument('-n', '--name', default=None, type=str)
    args = parser.parse_args()

    TESTDIR = "data/"
    model_ckpt_file = f'{args.name}.pt' if args.name else os.listdir("model_checkpoint")[-1]
    print(f"Using model checkpoint: {model_ckpt_file}")

    # Load the test dataset and model
    test_loader = DataLoader(TomatoLeafDataset(TESTDIR + "test.csv", TESTDIR + "test"), batch_size=1)
    # model = TomatoLeafModel()
    model = DoubleTomatoLeafModel()
    
    weights = torch.load("model_checkpoint/" + model_ckpt_file, weights_only=True)
    model.load_state_dict(weights)
    test_df = pd.read_csv(TESTDIR + "test.csv")

    # Create predictions for each of image and append it to the csv file
    for sample in test_loader:
        img = sample['image']
        id = sample['id'][0]
        print(f"Predicting for image: {id}")

        pred_mask = model(img)
        test_df.loc[test_df['id'] == id, 'annotation'] = encode_mask(pred_mask.detach(), 0.9)
        # test_df['annotation'] = encode_mask(pred_mask.detach(), 0.9)

    output_path = f"predictions/{args.name}" if args.name else "predictions"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    test_df.to_csv(f"predictions/{args.name}/{args.name}_submission.csv")

    if args.image:
        images, masks = load_and_decode(f"predictions/{args.name}/{args.name}_submission.csv", TESTDIR + "test")
        for i in range(len(images)):
            name = test_df.iloc[i]['id']
            display_image_and_mask(images[i], masks[i], name, f"predictions/{args.name}/images")

           

if __name__ == "__main__":
    main()