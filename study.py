import os
import nltk
import pickle
import numpy as np
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import ResNet152_Weights


class Vocab(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0

    def __call__(self, token):
        if not token in self.w2i:
            return self.w2i["<unk>"]
        return self.w2i[token]

    def __len__(self):
        return len(self.w2i)

    def add_token(self, token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1


def build_vocabulary(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]["caption"])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    tokens = [token for token, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocab()
    vocab.add_token("<pad>")
    vocab.add_token("<start>")
    vocab.add_token("<end>")
    vocab.add_token("<unk>")

    # Add the words to the vocabulary.
    for i, token in enumerate(tokens):
        vocab.add_token(token)
    return vocab


class CustomCocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, data_path, coco_json_path, vocabulary, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = data_path
        self.coco_data = COCO(coco_json_path)
        self.indices = list(self.coco_data.anns.keys())
        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self, idx):
        """Returns one data pair (image and caption)."""
        coco_data = self.coco_data
        vocabulary = self.vocabulary
        annotation_id = self.indices[idx]
        caption = coco_data.anns[annotation_id]["caption"]
        image_id = coco_data.anns[annotation_id]["image_id"]
        image_path = coco_data.loadImgs(image_id)[0]["file_name"]

        image = Image.open(os.path.join(self.root, image_path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocabulary("<start>"))
        caption.extend([vocabulary(token) for token in word_tokens])
        caption.append(vocabulary("<end>"))
        ground_truth = torch.Tensor(caption)
        return image, ground_truth

    def __len__(self):
        return len(self.indices)


def collate_function(data_batch):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    imgs, caps = zip(*data_batch)

    # Merge images (from list of 3D tensors to 4D tensor).
    # Originally, imgs is a list of <batch_size> number of RGB images with dimensions (3, 256, 256)
    # This line of code turns it into a single tensor of dimensions (<batch_size>, 3, 256, 256)
    imgs = torch.stack(imgs, 0)

    # Merge captions (from list of 1D tensors to 2D tensor), similar to merging of images donw above.
    cap_lens = [len(cap) for cap in caps]
    tgts = torch.zeros(len(caps), max(cap_lens)).long()
    for i, cap in enumerate(caps):
        end = cap_lens[i]
        tgts[i, :end] = cap[:end]
    return imgs, tgts, cap_lens


def get_loader(
    data_path, coco_json_path, vocabulary, transform, batch_size, shuffle, num_workers
):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco_dataser = CustomCocoDataset(
        data_path=data_path,
        coco_json_path=coco_json_path,
        vocabulary=vocabulary,
        transform=transform,
    )

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    custom_data_loader = torch.utils.data.DataLoader(
        dataset=coco_dataser,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_function,
    )
    return custom_data_loader


class CNNModel(nn.Module):
    def __init__(self, embedding_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNNModel, self).__init__()
        resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        module_list = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet_module = nn.Sequential(*module_list)
        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        # self.layer_norm = nn.LayerNorm(embedding_size)

    def forward(self, input_images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            resnet_features = self.resnet_module(input_images)
        resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
        linear_features = self.linear_layer(resnet_features)
        # final_features = self.layer_norm(linear_features)
        # return final_features
        return linear_features


class LSTMModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_layer_size,
        vocabulary_size,
        num_layers,
        max_seq_len=20,
    ):
        """Set the hyper-parameters and build the layers."""
        super(LSTMModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_layer = nn.LSTM(
            embedding_size, hidden_layer_size, num_layers, batch_first=True
        )
        self.linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_features, capts, lens):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embedding_layer(capts)
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        lstm_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        hidden_variables, _ = self.lstm_layer(lstm_input)
        model_outputs = self.linear_layer(hidden_variables[0])
        return model_outputs

    def sample(self, input_features, lstm_states=None):
        """Generate captions for given image features using greedy search."""
        sampled_indices = []
        lstm_inputs = input_features.unsqueeze(1)
        for i in range(self.max_seq_len):
            hidden_variables, lstm_states = self.lstm_layer(
                lstm_inputs, lstm_states
            )  # hiddens: (batch_size, 1, hidden_size)
            model_outputs = self.linear_layer(
                hidden_variables.squeeze(1)
            )  # outputs:  (batch_size, vocab_size)
            _, predicted_outputs = model_outputs.max(1)  # predicted: (batch_size)
            sampled_indices.append(predicted_outputs)
            lstm_inputs = self.embedding_layer(
                predicted_outputs
            )  # inputs: (batch_size, embed_size)
            lstm_inputs = lstm_inputs.unsqueeze(
                1
            )  # inputs: (batch_size, 1, embed_size)
        sampled_indices = torch.stack(
            sampled_indices, 1
        )  # sampled_ids: (batch_size, max_seq_length)
        return sampled_indices


def train(
    encoder_model, decoder_model, loss_criterion, optimizer, custom_data_loader, device
):
    total_num_steps = len(custom_data_loader)
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for i, (imgs, caps, lens) in enumerate(custom_data_loader):

            # Set mini-batch dataset
            imgs = imgs.to(device)
            caps = caps.to(device)
            tgts = pack_padded_sequence(caps, lens, batch_first=True)[0]

            # Forward, backward and optimize
            feats = encoder_model(imgs)
            outputs = decoder_model(feats, caps, lens)
            loss = loss_criterion(outputs, tgts)
            decoder_model.zero_grad()
            encoder_model.zero_grad()
            loss.backward()
            optimizer.step()

            stats = "Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f" % (
                epoch,
                num_epochs,
                i,
                total_num_steps,
                loss.item(),
                np.exp(loss.item()),
            )
            print("\r" + stats, end="")

    torch.save(decoder_model.state_dict(), os.path.join("", "decoder-lstm.pkl"))
    torch.save(encoder_model.state_dict(), os.path.join("", "encoder-lstm.pkl"))


def denormalize(img, mean, std):
    img = img * std[None, None, :] + mean[None, None, :]
    return img


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    isTrain = False
    isInit = False
    batch_size = 32
    learning_rate = 0.001
    image_size = 256
    embed_size = 256
    hidden_size = 512

    if isInit:
        nltk.download("punkt")

        vocab = build_vocabulary(
            json="coco/annotations/captions_train2014.json", threshold=4
        )
        vocab_path = "coco/vocab.pkl"
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        print("Total vocabulary size: {}".format(len(vocab)))
        print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

    if isTrain:

        # Load vocabulary wrapper
        with open("coco/vocab.pkl", "rb") as f:
            vocabulary = pickle.load(f)

        # Image preprocessing, normalization for the pretrained resnet
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # # Build data loader
        custom_data_loader = get_loader(
            "coco/train2014",
            "coco/annotations/captions_train2014.json",
            vocabulary,
            transform,
            batch_size,
            shuffle=True,
            num_workers=2,
        )

        # Build the models
        encoder_model = CNNModel(embed_size).to(device)
        decoder_model = LSTMModel(embed_size, hidden_size, len(vocabulary), 1).to(
            device
        )

        # Loss and optimizer
        loss_criterion = nn.CrossEntropyLoss()
        parameters = (
            list(decoder_model.parameters())
            + list(encoder_model.linear_layer.parameters())
            # + list(encoder_model.layer_norm.parameters())
        )
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        train(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            loss_criterion=loss_criterion,
            optimizer=optimizer,
            custom_data_loader=custom_data_loader,
            device=device,
        )
    else:

        with open("coco/vocab.pkl", "rb") as f:
            vocabulary = pickle.load(f)

        transform_val = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

        custom_data_loader = get_loader(
            "coco/val2014",
            "coco/annotations/captions_val2014.json",
            vocabulary,
            transform_val,
            batch_size,
            shuffle=True,
            num_workers=2,
        )

        # Build models
        encoder_model = CNNModel(
            embed_size
        ).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder_model = LSTMModel(embed_size, hidden_size, len(vocabulary), 1).eval()
        encoder_model = encoder_model.to(device)
        decoder_model = decoder_model.to(device)

        # Load the trained model parameters
        encoder_model.load_state_dict(torch.load("encoder-lstm.pkl"))
        decoder_model.load_state_dict(torch.load("decoder-lstm.pkl"))

        orig_images, images, lengths = next(iter(custom_data_loader))

        # Choose the first image in the batch
        orig_image = orig_images[0].to(
            device
        )  # Ensure the image is on the correct device

        # Prepare the image for display
        img = orig_image.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = denormalize(img, mean, std)
        img = np.clip(img, 0, 1)

        # Add batch dimension and move to device
        orig_image = orig_image.unsqueeze(0).to(device)

        features = encoder_model(orig_image)
        output = decoder_model.sample(features)

        predicted_caption = []
        for token_index in output[0]:  # Use the first element from the batch
            word = vocabulary.i2w[token_index.item()]
            predicted_caption.append(word)
            if word == "<end>":
                break
        predicted_sentence = " ".join(predicted_caption)

        # Plot the image
        plt.imshow(img)
        plt.title(predicted_sentence)
        plt.show()
