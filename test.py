import argparse
import os
import pandas as pd
import torch
from datasets import *
# Use the new Shake-Shake model for testing
from modresnet import ShakeShakeResNet26
import torchvision
from all_transforms import *
from utils import progress_bar


def passTestData(model, device, state_dict, testloader, nolabel=False):
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        all_ids = []
        all_predictions = []
        all_labels = []

        for batch_idx, batch_data in enumerate(testloader):
            images = batch_data[0].to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            if nolabel:
                labels = None
                ids = batch_data[1].to(device)
                all_ids += ids.tolist()
                all_predictions += predicted.tolist()
            else:
                labels = batch_data[1].to(device)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_labels += labels.tolist()
                progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                             % (100. * correct / total, correct, total))

    return all_ids, all_predictions, correct, total, all_labels


def testModel(exp_name, ckpt_name, testloader=None, model=None, device='cpu', nolabel=False):
    checkpoint = torch.load(os.path.join(exp_name, exp_name + '_checkpoints', exp_name + '_' + ckpt_name),
                            map_location=device)

    test_accuracy = checkpoint.get('acc', 'unknown')
    test_epoch = checkpoint.get('epoch', 'unknown')

    print('CIFAR-10 test accuracy: {}% at epoch {}'.format(test_accuracy, test_epoch))

    if testloader is not None:
        if nolabel:
            all_ids, all_predictions, _, _, _ = passTestData(model, device, checkpoint.get('net'), testloader, nolabel)
            df = pd.DataFrame({'ID': all_ids, 'Labels': all_predictions})
            df = df.sort_values(by='ID')
            df.to_csv(os.path.join(exp_name, 'submission.csv'), index=False)
            print('submission.csv generated!')
        else:
            _, _, correct, total, _ = passTestData(model, device, checkpoint.get('net'), testloader, nolabel)
            test_accuracy = 100. * correct / total
            print('Test accuracy of given dataset: {}% at epoch {}'.format(test_accuracy, test_epoch))

    return test_accuracy, test_epoch


def testModelTTA(exp_name, ckpt_name, model, device='cpu', nolabel=False):
    checkpoint = torch.load(os.path.join(exp_name, exp_name + '_checkpoints', exp_name + '_' + ckpt_name),
                            map_location=device)
    test_epoch = checkpoint.get('epoch', 'unknown')
    model.load_state_dict(checkpoint['net'])
    model.eval()

    all_predictions = []

    for i in range(len(tta_transforms)):
        all_outputs = []
        if args.nolabel:
            tta_testset = CustomCIFAR10Dataset(root='./data', mode='test_nolabel',
                                               pkl_file_path='cifar_test_nolabel.pkl', transform=tta_transforms[i])
            if i == 0:
                all_ids = tta_testset.return_ids().tolist()
        else:
            tta_testset = CustomCIFAR10Dataset(root='./data', mode='test', transform=tta_transforms[i])
            if i == 0:
                all_labels = tta_testset.return_labels().tolist()
        tta_testloader = normalized_loader(tta_testset, n=5, batch_size=100, shuffle=False, num_workers=2)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tta_testloader):
                images = batch_data[0].to(device)
                outputs = model(images)
                all_outputs.append(outputs)

        all_predictions.append(torch.cat(all_outputs))
        print('{}/{} transforms'.format(i + 1, len(tta_transforms)))

    final_probs = sum(all_predictions) / len(all_predictions)
    predicted = torch.argmax(final_probs, dim=1)

    if nolabel:
        df = pd.DataFrame({'ID': all_ids, 'Labels': predicted.tolist()})
        df = df.sort_values(by='ID')
        df.to_csv(os.path.join(exp_name, 'submission.csv'), index=False)
        print('submission.csv generated!')
        test_accuracy = None
    else:
        correct = predicted.eq(torch.tensor(all_labels).to(device)).sum().item()
        test_accuracy = 100. * correct / len(tta_testset)
        print('TTA based test accuracy of given dataset: {}% at epoch {}'.format(test_accuracy, test_epoch))

    return test_accuracy, test_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing a model')
    parser.add_argument('--exp_name', default='exp_shakeShake', type=str, help='Experiment name')
    parser.add_argument('--ckpt_name', default='best.pth', type=str, help='Checkpoint')
    parser.add_argument('--nolabel', default=0, type=int, help='Generate CSV for unlabeled data')
    parser.add_argument('--tta', default=0, type=int, help='Do test-time augmentation')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Build the model using the new Shake-Shake ResNet-26
    net = ShakeShakeResNet26()
    net = net.to(device)

    if args.nolabel:
        testset = CustomCIFAR10Dataset(root='./data', mode='test_nolabel', pkl_file_path='cifar_test_nolabel.pkl',
                                       transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    else:
        testloader = None

    if args.tta:
        _, _ = testModelTTA(args.exp_name, args.ckpt_name, net, device=device, nolabel=args.nolabel)
    else:
        _, _ = testModel(args.exp_name, args.ckpt_name, testloader, net, device=device, nolabel=args.nolabel)
