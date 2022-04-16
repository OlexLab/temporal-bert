# Amy Olex
# Training Clinical BERT prediction on classifying temporal token types using the BIO this_model.
# In this file I am attempting to fine-tune Clinical BERT to predict specific temporal expressions and their types.
#
# MOST of this code came straight from the Fine-Tuning BERT Tutorial by Chris McCormick at
# https://mccormickml.com/2019/07/22/BERT-fine-tuning/.

# Will need to reference this migration checklist to change the BertAdam stuff: https://huggingface.co/transformers/migration.html
import argparse
import itertools

import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AutoTokenizer
from transformers import AdamW, BertForTokenClassification, AutoModelForSequenceClassification
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import seaborn as sns
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv





# Steps to setting up Pretraining
#
# 1) enable GPU
# 2) import dataset and prep sentences for BERT (DONE)
# 3) Load pre-trained BERT model_bert for Tokenizer (DONE)
# 4) Tokenize Text to get BERT tokens, padding, and masks as prep for input to trainer (DONE)
# 5) Split dataset into train and test. Include option for ablation. (DONE)
# 6) Convert into Torch Tensors and TensorDatasets for sending to GPU. (DONE)
# 7) Set up hyper parameters and train model_bert.
# 8) Assess training evaluation
# 9) Generate predictions on test set
# 10) Evaluate performance on test set


def addGPU():
    # Add a_bert GPU accelerator
    # Not sure if this will work on Pine
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    # specify the GPU name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)


def load_tokenizer(filepath):
    """
    Initializes and loads the BERT pre-trained tokenizer_bert.
    :param filepath: The path and file name of the pretrained BERT model_bert to use for tokenizing.
    :return tokenizer_bert: Returns a_bert BERT tokenizer_bert object.
    """
    # Identify the pretrained Clinical BERT model_bert
    # Will need to download this and upload to Pine I think
    tok = BertTokenizer.from_pretrained(filepath, do_lower_case=True)
    # tokenizer_bert = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return tok


'''def tokenizeText(tokeizer):
    tokenized_texts = [tokenizer_bert.tokenize(sent) for sent in sentences]
    print("Tokenize the first sentence:")
    print(tokenized_texts[0])

    # Set the maximum sequence length. The longest sequence in our training set is 502, but we'll leave room on the end anyway.
    # In the original paper, the authors used a_bert length of 512, so we will too.
    MAX_LEN = 0
    for s in sentences:
        if len(s) > MAX_LEN:
            MAX_LEN = len(s)
    MAX_LEN

    # Prepare inputs, create train and test sets
    # This can be turned into a_bert function
    MAX_LEN = 512

    # Pad our input tokens
    input_ids = pad_sequences([tokenizer_bert.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Use the BERT tokenizer_bert to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer_bert.convert_tokens_to_ids(x) for x in tokenized_texts]

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a_bert mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
'''


def splitTrainTest(sentences, labels, test_size=0.1):  # input_ids, labs, attention_masks, test_size=0.1, ablation = 1):
    # Use train_test_split to split our data into train and validation sets for training

    train_sents, validation_sents, train_labels, validation_labels = train_test_split(sentences, labels,
                                                                                      random_state=42,
                                                                                      test_size=test_size)

    # train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labs,
    #                                                                                    random_state=42,
    #                                                                                    test_size=test_size)
    # train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=42,
    #                                                       test_size=test_size)

    # return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks
    return train_sents, validation_sents, train_labels, validation_labels


def createDataLoader(inputs, masks, tags, batch_size=16):
    # Convert all data into tensors and do more formatting (again, put in a_bert function)
    # Convert all of our data into torch tensors, the required datatype for our model_bert

    print(inputs[0])
    inputs = torch.stack(inputs).squeeze()

    labels = torch.tensor(tags).squeeze()

    masks = torch.stack(masks).squeeze()

    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return data, sampler, dataloader


## This function came from: https://www.topbots.com/fine-tune-transformers-in-pytorch/
def train(dataloader, optimizer, scheduler, device):
    r"""
    Train pytorch model_bert on a_bert single pass through the data loader.

    It will use the global variable `model_bert` which is the transformer model_bert
    loaded on `_device` that we want to train on.

    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a_bert batch in dictionary format that can be passed
      straight into the model_bert - `model_bert(**batch)`.

    Arguments:

        dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

        optimizer_ (:obj:`transformers.optimization.AdamW`):
            Optimizer used for training.

        scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
            PyTorch scheduler.

        device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model_bert.

    Returns:

        :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss].
    """

    # Use global variable for model_bert.
    global model

    # Tracking variables.
    predictions_labels = []
    true_labels = []
    # Total loss for this epoch.
    total_loss = 0

    # Put the model_bert into training mode.
    model.train()

    # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):
        # Add original labels - use later for evaluation.

        true_labels += batch['Labels'].numpy().flatten().tolist()

        # move batch to device
        # batch is a_bert dict.  what is the k, v?
        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}
        print("My Batch:")
        print(batch)
        # Always clear any previously calculated gradients before performing a_bert
        # backward pass.
        model.zero_grad()

        # Perform a_bert forward pass (evaluate the model_bert on this training batch).
        # This will return the loss (rather than the model_bert output) because we
        # have provided the `labels`.
        # The documentation for this a_bert bert model_bert function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model_bert` always returns a_bert tuple, so we need to pull the
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a_bert Tensor containing a_bert
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a_bert backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a_bert step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


## This function was copied from https://www.topbots.com/fine-tune-transformers-in-pytorch/
def validation(dataloader, device):
    r"""Validation function to evaluate model_bert performance on a_bert
    separate set of data.

    This function will return the true and predicted labels so we can use later
    to evaluate the model_bert's performance.

    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a_bert batch in dictionary format that can be passed
      straight into the model_bert - `model_bert(**batch)`.

    Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

      device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model_bert.

    Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss]
    """

    # Use global variable for model_bert.
    global model

    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    # Put the model_bert in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device) for k, v in batch.items()}

        # Telling the model_bert not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model_bert` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)

            # The call to `model_bert` always returns a_bert tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            loss, logits = outputs[:2]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a_bert Tensor containing a_bert
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            # update list
            predictions_labels += predict_content

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def train2(train_dataloader, optimizer, scheduler):
    global model
    # Set our model_bert to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    # Total loss for this epoch.
    tl_set = []
    total_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        print("Step: " + str(step))
        # print("Batch: ")
        # print(batch)
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # print("Len Batch InputIDs: " + str(len(b_input_ids)))
        # print("Len Batch[0] InputIDs: " + str(len(b_input_ids[0])))

        # print("Len Mask InputIDs: " + str(len(b_input_mask)))
        # print("Len Mask[0] InputIDs: " + str(len(b_input_mask[0])))

        # print("Len Labels InputIDs: " + str(len(b_labels)))
        # print("Len Labels[0] InputIDs: " + str(len(b_labels[0])))

        # Clear out the gradients (by default they accumulate)
        # optimizer.zero_grad()
        model.zero_grad()

        # Forward pass
        # print("Forward pass")
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        loss, logits = outputs[:2]
        # print("Loss:")
        # print(loss)
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a_bert Tensor containing a_bert
        # single value; the `.item()` function just returns the Python value
        # from the tensor.

        tl_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a_bert step using the computed gradient
        optimizer.step()
        scheduler.step()

        # Update tracking variables
        total_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    avg_train_loss = total_loss / nb_tr_steps
    print("Length: " + str(len(tl_set)))
    print("Average total train loss: {}".format(total_loss / nb_tr_steps))
    print("Total Loss for this epoch: " + str(total_loss))
    print("Number of steps for this epoch: " + str(nb_tr_steps))
    return tl_set, avg_train_loss


def fineTuneModel(model, train_dataloader, validation_dataloader):
    # Load BertForSequenceClassification, the pretrained BERT model_bert with a_bert single linear classification layer on top.
    # AdamW information is here: https://huggingface.co/transformers/migration.html

    # Store our loss and accuracy for plotting
    train_loss_set = []
    torch.device.empty_cache()

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 2

    # trange is a_bert tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

        # Training

        # Set our model_bert to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a_bert step using the computed gradient
            optimizer.step()
            scheduler.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put model_bert in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model_bert not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))


def assessTraining(train_loss_set):
    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.savefig('figs/train_loss_set.png')
    # plt.show()


def calcPRF1(true_labels, predictions):
    # Import and evaluate each test batch using Matthew's correlation coefficient
    from sklearn.metrics import matthews_corrcoef
    matthews_set = []

    for i in range(len(true_labels)):
        matthews = matthews_corrcoef(true_labels[i],
                                     np.argmax(predictions[i], axis=1).flatten())
        matthews_set.append(matthews)

    # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    matthews_corrcoef(flat_true_labels, flat_predictions)

    pred = [1 if x == 0 else 0 for x in flat_predictions]
    np.array(pred)

    pred = [1 if x == 1 else 0 for x in flat_predictions]
    np.asarray(pred)

    labels = [1 if x == 1 else 0 for x in flat_true_labels]
    np.array(labels)

    np.asarray([1 if (str(x) + str(y)) == "10" else 0 for x, y in zip(labels, pred)])

    groups = np.unique(flat_true_labels)

    results = []

    for g in groups:
        pred = [1 if x == g else 0 for x in flat_predictions]
        labels = [1 if x == g else 0 for x in flat_true_labels]

        TP = sum([1 if (x + y) == 2 else 0 for x, y in zip(labels, pred)])
        TN = sum([1 if (x + y) == 0 else 0 for x, y in zip(labels, pred)])
        FP = sum([1 if (str(x) + str(y)) == "01" else 0 for x, y in zip(labels, pred)])
        FN = sum([1 if (str(x) + str(y)) == "10" else 0 for x, y in zip(labels, pred)])

        if TP == 0:
            P = "NA"
            R = "NA"
            F1 = "NA"
        else:
            P = round(TP / (TP + FP), 3)
            R = round(TP / (TP + FN), 3)
            F1 = round(2 * ((P * R) / (P + R)), 3)

        results.append([P, R, F1])
        print("For Label " + str(g) + " TP=" + str(TP) + " TN=" + str(TN) + " FP=" + str(FP) + " FN=" + str(
            FN) + "; P=" + str(P) + " R=" + str(R) + " F1=" + str(F1))

    print("Challenge\t\tP\tR\tF1\n")
    for (p, r, f), y in zip(results,
                            ["NA", "Confidence\t", "Overwhelmed\t", "Systems Issues\t", "Supportive Environment"]):
        if y != "NA":
            print(y + "\t" + str(p) + "\t" + str(r) + "\t" + str(f))


def calcMCC(true_labels, predictions):
    matthews_set = []

    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating Matthews Corr. Coef. for each batch...')

    # For each input batch...
    for i in range(len(true_labels)):
        # The predictions for this batch are a_bert 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a_bert list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()

        # Calculate and store the coef for this batch.
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)

    # Create a_bert barplot showing the MCC score for each batch of test samples.
    ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

    plt.title('MCC Score per Batch')
    plt.ylabel('MCC Score (-1 to +1)')
    plt.xlabel('Batch #')

    plt.savefig('figs/MCC.png')

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a_bert single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

    return mcc
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def predict(dataloader):
    # Validation

    # Put model_bert in evaluation mode to evaluate loss on the validation set
    global model
    model.eval()

    # Tracking variables
    total_eval_loss, total_eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Tracking variables
    predictions, true_labels = [], []

    # Evaluate data for one epoch
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model_bert not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits
            # loss = outputs.loss

        # Accumulate the validation loss.
        # I can't get this to work because the Loss comes back as a_bert None type.
        # total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

        # eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

        # get predicitons to list
        predict_content = logits.argmax(axis=-1).flatten().tolist()

        # Store predictions and true labels
        predictions.append(predict_content)
        true_labels.append(label_ids)

    print("Validation Accuracy: {}".format(total_eval_accuracy / nb_eval_steps))

    return total_eval_accuracy, predictions, true_labels  # , total_eval_loss

def loadData(filename, has_labels=True, level="", sample_size=1.0):
    # import dataset for training
    if level == "token":
        f = open(filename)
        lines = f.read().splitlines()
        f.close()

        if has_labels:
            s = list(split_at([l.split(' ')[0] if l else '' for l in lines], lambda x: not x))
            l = list(split_at([l.split(' ')[1] if l else '' for l in lines], lambda x: not x))

        else:
            s = list(split_at([l.split(' ')[0] if l else '' for l in lines], lambda x: not x))
            l = ""

    else:
        if has_labels:
            df = pd.read_csv(filename, delimiter='\t', header=None, names=['sent', 'label'])

            if sample_size < 1.0:
                num_records = floor(len(df) * sample_size)
                df = df.groupby('label').apply(lambda x: x.sample(num_records))
                print("Sub setting input to " + str(num_records))

            s = df.sent.values
            l = df.label.values
        else:
            df = pd.read_csv(filename, delimiter='\t', header=None, names=['sent'])

            if sample_size < 1.0:
                num_records = floor(len(df) * sample_size)
                df = df.apply(lambda x: x.sample(num_records))
                print("Sub setting input to " + str(num_records))

            s = df.sent.values
            l = ""

    return s, l


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


if __name__ == '__main__':

    ## Parse input arguments
    parser = argparse.ArgumentParser(
        description='Fine tune a BERT this_model on the seq2seq temporal task.')
    parser.add_argument('-m', metavar='modelname', type=str,
                        help='Name/path of the starting BERT this_model.',
                        required=True)
    parser.add_argument('-d', metavar='traindata', type=str,
                        help='Name of the data set to use for training and validation.',
                        required=True)
    parser.add_argument('-e', metavar='evaldata', type=str,
                        help='Name of the dataset to do final evaluation.',
                        required=True)
    parser.add_argument('-o', metavar='outputdir', type=str,
                        help='Path to the output directory where finetuned this_model should be saved. Default is current working directory.',
                        required=False, default='./')
    parser.add_argument('-p', metavar='epochs', type=int,
                        help='The number of epochs to fine tune over.', required=False, default=2)
    parser.add_argument('-s', metavar='batchsize', type=int,
                        help='Number of observations to pass to the GPU at the same time. Default is 16',
                        required=False, default=16)
    parser.add_argument('-l', metavar='maxlength', type=int,
                        help='The maximum sentence length to pad or trim too.  Default is 256.',
                        required=False, default=256)
    parser.add_argument('-c', metavar='cudadevice', type=str,
                        help='Name of cuda device to use. Default is 0.',
                        required=False, default="0")
    parser.add_argument('-t', metavar='tag2idx', type=str,
                        help='File with list of tags and their indexes.',
                        required=True)

    args = parser.parse_args()

    bert_model_path = args.m  #'../models/biobert_pretrain_output_all_notes_150000'
    #dev_data = "data/mini_timex.tsv"
    #train_data = "data/train_timex.tsv"
    test_data = args.e  #"data/test_timex.tsv"
    #mini_data = "data/mini_timex.tsv"
    training_data = args.d  #train_data
    #starting_bert_model = "bert_base"
    #starting_bert_model = "clin_bert"
    outfile = args.o  #"models/bert2chrono_seq2seq_pretrained"


    # Number of training epochs (authors recommend between 2 and 4)
    epochs = args.p  #2
    # Number of sentences to pass to GPU at a_bert time
    batch_size = args.s  #32
    max_length = args.l  #256
    max_grad_norm = 1.0
    # labels_ids = {'neg': 0, 'pos': 1}
    # n_labels = len(labels_ids)

    # Set up the GPU environment

    os.environ["CUDA_VISIBLE_DEVICES"] = args.c  #"0"
    # tf_device = '/gpu:0'

    # 1) Import Pre-trained bert model_bert with parameters and set the device to cpu or gpu if avaliable
    # initialize tokenizer_bert                                      num_labels=n_labels)

    # Get model_bert's tokenizer_bert.
    print('Loading tokenizer_bert...')

    #if starting_bert_model == "bert_base":
    #    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #else:
    tokenizer = load_tokenizer(bert_model_path)


    # tokenizer = load_tokenizer(bert_model_path)
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer_bert = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

    # exit(0)

    print("preprocessing input...")
    ### 2) Import and format dataset
    sentences, labels = loadData(training_data, level="token")

    ### get enumerated tag list
    with open(args.t, mode='r') as infile:
        reader = csv.reader(infile)
        tag2idx = {rows[0]: int(rows[1]) for rows in reader}

    # tags = list(set([item for sublist in labels for item in sublist]))
    # tags.append("PAD")
    # tag2idx = {t: i for i, t in enumerate(tags)}

    # print("Tags in Training Data: " + str(tag2idx))

    ### Tokenize and format into 2 lists of lists.
    tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs, tokenizer)
                                  for sent, labs in zip(sentences, labels)]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    ### Pad sentences
    input_ids = torch.tensor(pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                           maxlen=max_length, dtype="long", value=0.0,
                                           truncating="post", padding="post"))

    print(labels[1:3])
    print([[tag2idx.get(l) for l in lab] for lab in labels][1:3])
    tags = torch.tensor(pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                                      maxlen=max_length, value=tag2idx["PAD"], padding="post",
                                      dtype="long", truncating="post"))

    ### create attention masks
    attention_masks = torch.tensor([[float(i != 0.0) for i in ii] for ii in input_ids])

    print("splitting to train and test...")
    # Split into train and test datasets
    # I may not need to do this for the i2b2 data as it already have train and dev datasets
    train_inputs, validation_inputs, train_labels, validation_labels = splitTrainTest(input_ids, tags)
    train_masks, validation_masks, _, _ = splitTrainTest(attention_masks, input_ids)
    # format, tokenize, and tensorfy sentences and labels.

    print("Length of Train Inputs: " + str(len(train_inputs)))
    print("Length of Validation Inputs: " + str(len(validation_inputs)))
    print("Length of Train Labels: " + str(len(train_labels)))
    print("Length of Validation Labels: " + str(len(validation_labels)))
    print("Length of Train Masks: " + str(len(train_masks)))
    print("Length of Validation Masks: " + str(len(validation_masks)))

    # Create DataLoaders for each set of data
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    print("train_inputs tensor Shape: " + str(train_inputs.shape))
    # print(train_data[0])

    # train_data, train_sampler, train_dataloader = createDataLoader(train_inputs, train_masks, train_labels, batch_size)
    # validation_data, validation_sampler, validation_dataloader = createDataLoader(validation_inputs, validation_masks,
    #                                                                              validation_labels, batch_size)

    # train_dataloader = DataLoader(train_encoded_dict, batch_size=2, shuffle=True, drop_last=True)
    # print("My train_dataloader:")
    # print(type(train_dataloader))
    # train_features, train_labels = train_dataloader.item(0)
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    print('Created `train_dataloader` with %d batches!' % len(train_dataloader))

    # Get the actual model_bert.
    print('Loading model_bert...')

    #if starting_bert_model == "bert_base":
    #    this_model = BertForTokenClassification.from_pretrained(
    #        "bert-base-uncased",
    #        num_labels=len(tag2idx),
    #        output_attentions=True,
    #        output_hidden_states=True
    #    )
    #else:
    model = BertForTokenClassification.from_pretrained(
            bert_model_path,
            num_labels=len(tag2idx),
            output_attentions=True,
            output_hidden_states=True,
            local_files_only = True
        )

    # Load the model_bert on the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Model loaded to `%s`' % device)

    ### 3) Import/Set up parameters for fine-tuning
    # Set up hyper parameters
    # Now we are getting the hyperparamaters suggested by the authors.
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs

    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = 0
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    # optimizer = AdamW(model_bert.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    ## 4) Fine-tune the model_bert
    # The following training code is copied from: https://www.topbots.com/fine-tune-transformers-in-pytorch/
    # Store the average loss after each epoch so we can plot them.
    # Store our loss and accuracy for plotting
    train_loss_set = []
    # torch.device.empty_cache()

    # We'll store a_bert number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # trange is a_bert tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        print("Epoch: ", _)
        print("run training")
        tls, avg_train_loss = train2(train_dataloader, optimizer, scheduler)
        print("Length of TLS: " + str(len(tls)))
        print(type(tls))
        train_loss_set = train_loss_set + tls

        print("run validation")
        total_eval_accuracy, predictions, true_labels = predict(valid_dataloader)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(valid_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        # can't get the loss from the eval model_bert.  See validation2() notes.
        # avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        # validation_time = format_time(time.time() - t0)

        # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        # print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': _ + 1,
                'Training Loss': avg_train_loss,
                # 'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                # 'Training Time': training_time,
                # 'Validation Time': validation_time
            }
        )

    print(str(len(train_loss_set)))
    assessTraining(train_loss_set)

    #### In the McCormick tutorial there are plots to show the training loss vs validation loss.
    #### I'm not sure why I can't get the validation loss, so will need to investigate at some point.
    #### All the plots and tables in the McCormick tutorial was skipped.
    print("Training complete!")

    model.save_pretrained(outfile)
    tokenizer.save_pretrained(outfile)

