import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.naive_nn import NaiveNN
from preprocessing.preprocess_data import CustomDataset


def learning():
    DATASET = CustomDataset()

    # TODO:
    #  - implement arg parser and obtain all the consts from console

    DATA_SIZE, FEATURES_SIZE = DATASET.shape
    BATCH_SIZE = 500
    H_DIMS = 200

    print(f"Amount of data read: {DATA_SIZE}")
    BATCHES = DataLoader(dataset=DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    # BATCHES = DataLoader(dataset=DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    print('Creating batches done')

    MODEL = NaiveNN(DATA_SIZE, BATCH_SIZE, FEATURES_SIZE, H_DIMS)
    EPOCHS = 100
    # ETA = 0.001
    ETA = 10
    # MOMENTUM = 0.9
    MOMENTUM = 0

    # CRITERION = nn.NLLLoss()
    CRITERION = nn.L1Loss()
    OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=ETA, momentum=MOMENTUM)

    NUN_BATCHES = np.ceil(DATA_SIZE / BATCH_SIZE)
    print(f"Upcoming batches: {NUN_BATCHES}")


    # MODEL.train()
    for epoch in range(EPOCHS):
        tick = time.time()
        loss_total = 0.0
        batch_count = 0

        for X, y in BATCHES:
            batch_count += 1
            naive_loss = CRITERION(MODEL(X), y)
            print(f"Batch {batch_count} loss: {naive_loss}, mean loss: {loss_total / batch_count}")
            loss_total += naive_loss.item()

            OPTIMIZER.zero_grad()
            naive_loss.backward()
            OPTIMIZER.step()

            # if not (batch_count) % 10 and batch_count > 0:
            #     print(f"Batch {batch_count} of epoch {epoch + 1}")
            #     print('mean loss {0:.4f}'.format(loss_total.item() / batch_count))

        tock = time.time()
        print("~~~~~~~~~~~~~~~~~~")
        print(f"Epoch: {epoch + 1} out of {EPOCHS}")
        print(f"Time per epoch: {tock- tick}")
        print(f"Mean loss: {loss_total / NUN_BATCHES}")
        print(f"Total loss: {loss_total}")
        print("~~~~~~~~~~~~~~~~~~")

        #  zapisywanie wag do pliku


if __name__ == "__main__":
    learning()