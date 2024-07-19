import datetime
import os

def save_args(args, rewrite=False):
    if args.MODEL.find('EffFcn') >= 0:
        save_path = os.path.join(os.path.join(args.SAVE_ROOT, args.MODEL), args.PRETRAINED_MODEL)
        save_path = os.path.join(save_path, args.DATASET)
    else:
        save_path = os.path.join(os.path.join(args.SAVE_ROOT, args.MODEL), args.DATASET)
    save_path = os.path.join(save_path, datetime.date.today().strftime('%Y.%m.%d'))
    save_path = os.path.join(save_path, args.SAVE_PREFIX)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, 'weight')):
        os.makedirs(os.path.join(save_path, 'weight'))
    with open(os.path.join(save_path, 'args.txt'), mode='w' if rewrite else 'a+') as f:
        f.write('{:^17}: {}\n'.format('CREATE_TIME', datetime.datetime.now()))
        f.write('----------------------------------------------------------\n')
        for arg in vars(args):
            f.write('{:^17}: {}\n'.format(arg, getattr(args, arg)))
        f.write('----------------------------------------------------------\n\n')
        f.close()
    return save_path