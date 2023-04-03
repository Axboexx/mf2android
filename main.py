"""
@Time : 2023/4/2 20:43
@Author : Axboexx
@File : main.py
@Software: PyCharm
"""
import datetime

from SAM.loss_function import *
from SAM.optimizer import *
from utils.utils2 import progress_bar
import model.mobileformer as mf
from utils.Dataset import data_prepare

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODULE_PATH = "E:\\A_A_data\\module_to_android\\mobileformer2android\\checkpoint"
EPOCH = 400
start_epoch = 0
batch_size_test = 20
batch_size_train = 20

print('==> Building model..')
net = mf.gernate_mf_294()
net.to(device)

if False:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('MODULE_PATH')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
base_optimizer = torch.optim.SGD
optimizer = SAM(net.parameters(), base_optimizer, rho=0.05, lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

trainloader, testloader = data_prepare(batch_size_train, batch_size_test)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    H = []
    Y = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        Y.append(targets)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = smooth_crossentropy(outputs, targets)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        smooth_crossentropy(net(inputs), targets).mean().backward()
        optimizer.second_step(zero_grad=True)

        train_loss += loss.mean().item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


        H.append(outputs.to('cpu'))
    H = torch.cat(H, 0).detach().numpy()
    Y = torch.cat(Y, 0).numpy()

    return train_loss / len(trainloader), 100. * correct / total, H, Y


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    H = []
    Y = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            Y.append(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = smooth_crossentropy(outputs, targets)
            # loss = criterion(outputs, targets)
            # loss = std_loss(outputs, targets)

            test_loss += loss.mean().item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            H.append(outputs.to('cpu'))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        now = datetime.datetime.now()
        torch.save(state, './checkpoint/' + f'sam_mf_{now:%Y%m%d}.pth')
        best_acc = acc

    H = torch.cat(H, 0).detach().numpy()
    Y = torch.cat(Y, 0).numpy()

    return test_loss / len(testloader), 100. * correct / total, H, Y


if __name__ == '__main__':
    tr_ls = []
    tr_as = []
    te_ls = []
    te_as = []

    for epoch in range(start_epoch, EPOCH):
        tr_l, tr_a, tr_H, tr_Y = train(epoch)
        te_l, te_a, te_H, te_Y = test(epoch)

        tr_ls.append(tr_l)
        tr_as.append(tr_a)
        te_ls.append(te_l)
        te_as.append(te_a)
        print(max(tr_as), "% ", max(te_as), "%")
        scheduler.step()
