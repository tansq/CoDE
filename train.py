import os

import numpy as np
import torch
torch.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import argparse
from mini_dataloader import DataLoader
from model import select_model
from state import select_state
from reward_function import basic_reward, bce_reward
from utils import cal_auc, cal_iou, cal_f1, AverageMeter, save_args, save_logs, Logger
from tqdm import tqdm
from GaussianA3C import GaussianA3C
import torch.optim as optim


#_/_/_/ training parameters _/_/_/
parser = argparse.ArgumentParser(description='forgery_detection_A3C')

# Basic parameters
parser.add_argument('--MODEL', type=str, default='AC_CoDE',
                    help='Training model.')
parser.add_argument('--PRETRAINED-MODEL', type=str, default='efficientnet-b4',
                    help='Pretrained model.')
parser.add_argument('--PRETRAINED', type=bool, default=True,
                    help='Pretrained model.')
parser.add_argument('--EXTERNAL-PRETRAINED', type=bool, default=False,
                    help='External Pretrained.')
parser.add_argument('--STATE', type=str, default='default',
                    help='State. default:(default)')
parser.add_argument('--NORMALIZE', type=bool, default=False,
                    help='Normalize the forgery image into [-1, 1]. default:(False)')
parser.add_argument('--DATASET', type=str, default='dataset',
                    help='Training dataset.')
parser.add_argument('--RATIO', type=float, default=0.7,
                    help='The ratio between train / test set.')
parser.add_argument('--RANDOM-DIVIDED', type=bool, default=True,
                    help='Randomly shuffle the whole dataset before divided. default:(True)')
parser.add_argument('--AUGMENT-PROB', type=float, default=0.2,
                    help='Probability of data augmentation(flip, rotation). default:(0.2)')
parser.add_argument('--INPUT-SIZE', type=int, default=512,
                    help='Size of image. default:(512)')
parser.add_argument('--LEARNING-RATE', type=float, default=0.0001,
                    help='Learning rate. default:(0.0001)')
parser.add_argument('--TRAIN-BATCH-SIZE', type=int, default=2,
                    help='The batch of training data. default:(8)')
parser.add_argument('--VAL-BATCH-SIZE', type=int, default=1,
                    help='The batch of val data. default:(1)')
parser.add_argument('--SAVE-ROOT', type=str, default='./weights',
                    help='The root of save path to the model.')
parser.add_argument('--SAVE-PREFIX', type=str, default='baseline',
                    help='The prefix of save path to the model.')

# Parameters of the reinforcement learning framework
parser.add_argument('--EPISODE-LEN', type=int, default=3,
                    help='Total experience of one episode. default:(3)')
parser.add_argument('--GAMMA', type=float, default=0.95,
                    help='Discount factor for future rewards. default:(0.95)')
parser.add_argument('--N-EPISODES', type=int, default=50000,
                    help='Total episodes. default:(50000)')
parser.add_argument('--SNAPSHOT-EPISODES', type=int, default=500,
                    help='The interval at which the model is saved. default:(500)')
parser.add_argument('--FORGERY-THRESHOLD', type=float, default=0.2,
                    help='Threshold of tampering. If the probability is greater than the threshold, '
                         'it is considered as tampering. default:(0.2)')


def val(save_path, episode, args, model, val_batch_loader):
    model.eval()

    # initialize state
    current_state = select_state(args.STATE)

    f1 = AverageMeter()
    auc = AverageMeter()
    iou = AverageMeter()
    sum_reward = AverageMeter()
    t = tqdm(range(len(val_batch_loader.flist)))
    for i in t:
        sum_external_reward = 0
        # load a batch of data
        forgery, gt_mask = val_batch_loader.test_get_item(i)
        # initialize the current state and reward
        current_state.reset(forgery)
        for j in range(0, args.EPISODE_LEN):
            with torch.no_grad():
                previous_mask = current_state.mask.copy()
                statevar = torch.Tensor(current_state.state).cuda()
                # ------- #
                if j == 0:
                    p, v = model.forgery_forward(statevar)
                mu, _, _ = model.prob_forward(statevar, p, v)
                # ------- #
                mu *= 0.5
                actions = mu.detach().cpu().numpy()
                current_state.step(actions)
                # reward
                reward = bce_reward(previous_mask, current_state.mask.copy(), gt_mask.copy())
                sum_external_reward += np.mean(reward) * np.power(args.GAMMA, j)
        sum_reward.update(sum_external_reward)
        # calculate metrics
        save_mask = current_state.mask.copy()
        save_gt_mask = gt_mask.copy()
        auc.update(cal_auc(save_mask, save_gt_mask))
        save_mask = np.where(save_mask > args.FORGERY_THRESHOLD, 1., 0)
        f1.update(cal_f1(save_mask, save_gt_mask))
        iou.update(cal_iou(save_mask, save_gt_mask))
        t.set_description('episode: {} | F1: {:.4f} | AUC: {:.4f} | IoU: {:.4f} | Reward: {:.4f}'.
                          format(episode, f1.avg, auc.avg, iou.avg, sum_reward.avg))
    message = 'episode: {} | F1: {:.4f} | AUC: {:.4f} | IoU: {:.4f} | Reward: {:.4f}'.\
        format(episode, f1.avg, auc.avg, iou.avg, sum_reward.avg)
    save_logs(save_path, message)
    model.train()
    return sum_reward.avg


def adjust_learning_rate(args, optimizer, episode):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = args.LEARNING_RATE * ((1-episode/args.N_EPISODES)**0.9)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def main():
    args = parser.parse_args()
    # _/_/_/ Save Parameter Settings _/_/_/
    save_path = save_args(args, rewrite=True)


    # _/_/_/ load training dataset _/_/_/
    train_batch_loader = DataLoader(name=args.DATASET, mode='train', ratio=args.RATIO, normalize=args.NORMALIZE,
                                    resize_shape=(args.INPUT_SIZE, args.INPUT_SIZE), batch_size=args.TRAIN_BATCH_SIZE,
                                    augment_prob=args.AUGMENT_PROB, divide_shuffle=args.RANDOM_DIVIDED,
                                    item_shuffle=False)
    # _/_/_/ load val dataset _/_/_/
    val_batch_loader = DataLoader(name=args.DATASET, mode='val', ratio=args.RATIO, normalize=args.NORMALIZE,
                                  resize_shape=(args.INPUT_SIZE, args.INPUT_SIZE), batch_size=args.VAL_BATCH_SIZE,
                                  augment_prob=0.0, divide_shuffle=args.RANDOM_DIVIDED, item_shuffle=False)
    print('Create dataloader: \033[93mdataset: {} | ratio: {}\033[0m'.format(args.DATASET, args.RATIO))

    # _/_/_/ initialize state _/_/_/
    current_state = select_state(args.STATE)
    print('Initialize state: \033[93m{}\033[0m'.format(args.STATE))

    # _/_/_/ load model _/_/_/
    model = select_model(args.MODEL).cuda()
    model.train()
    if args.EXTERNAL_PRETRAINED:
        LOAD_PATH = "./weights/pretrained_CoDE.pth"
        model.load_state_dict(torch.load(LOAD_PATH, map_location='cpu')['state_dict'])
        print('Load external pretrained weight: \033[93m{}\033[0m'.format(LOAD_PATH))
    print('Create shared model: \033[93m{}\033[0m'.format(args.MODEL))
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    agent = GaussianA3C(model, optimizer, args.TRAIN_BATCH_SIZE, args.EPISODE_LEN, args.GAMMA,
                        crop_size=args.INPUT_SIZE)

    train_logger = Logger(model_name=save_path, header=['epoch', 'lr', 'reward'])
    # _/_/_/ training _/_/_/
    for episode in range(1, args.N_EPISODES + 1):
        print("episode %d" % episode)
        # load a batch of data
        forgery, gt_mask = train_batch_loader.get_item()  # (b, 3, h, w), (b, 1, h, w)
        # initialize the current state and reward
        current_state.reset(forgery)
        reward = np.zeros(gt_mask.shape, gt_mask.dtype)  # (b, 1, h, w)
        sum_reward = 0
        for t in range(0, args.EPISODE_LEN):
            previous_mask = current_state.mask.copy()
            action = agent.act_and_train(current_state.state, reward, t)
            current_state.step(action)
            # reward
            reward = bce_reward(previous_mask, current_state.mask.copy(), gt_mask.copy())
            sum_reward += np.mean(reward) * np.power(args.GAMMA, t)
        agent.stop_episode_and_train(current_state.state, reward, False)
        print("Total Reward {:.4f}".format(sum_reward / args.EPISODE_LEN))

        if episode % args.SNAPSHOT_EPISODES == 0:
            reward = val(save_path, episode, args, model, val_batch_loader)
            # save train log
            train_logger.log(phase="train", values={
                'episode': episode,
                'lr': optimizer.param_groups[0]['lr'],
                'reward': reward
            })
            # save weight
            save_states_path = os.path.join(save_path, 'weight/{}.pth'.format(str(episode)))
            states = {
                'episode': episode,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            torch.save(states, save_states_path)

        adjust_learning_rate(args, optimizer, episode)


if __name__ == '__main__':
    main()
