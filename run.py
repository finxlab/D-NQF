#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

from train import *
# In[2]:

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

MODEL_FILENAME_DICT = {
    "DLinear": DLinear
    }

logger = logging.getLogger('Model.run')

parser = argparse.ArgumentParser()

# basic config
parser.add_argument('--random_seed', type = int, default=2025, help='Random Seed num')
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast', help='task name')

# data loader
parser.add_argument('--data', type = str, default='brw', help = 'dataset type')
parser.add_argument('--timeenc', action='store_false', help = 'Time encoder')
parser.add_argument('--scale', action='store_false', help = 'Target scale')
parser.add_argument('--model_type', type = str, default='DLinear', help='model type')
parser.add_argument('--model_name', type = str, default='model', help='Directory containing params.json')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--features', type=str, default='S',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='WIND SPEED', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly, ms:milliseconds], you can also use more detailed freq like 15min or 3h')

# forecasting task
parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# model define
parser.add_argument('--moving_avg', type=int, default=25, help='List for moving_avg window')
parser.add_argument('--n_quantiles', type=int, default=200, help='n_quantiles size')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='CRPS', help='loss function')


if __name__ == '__main__':

    configs = parser.parse_args()

    set_seed(configs.random_seed)
    Net = MODEL_FILENAME_DICT[configs.model_type]
    configs.model_dir = os.path.join(configs.model_name, configs.model_type, str(configs.pred_len), configs.data)
   
    if not os.path.exists(configs.model_dir):
        os.makedirs(configs.model_dir)

    configs.data_path = configs.data + '.csv'


    logger.info('Loading complete.')

    cuda_exist = torch.cuda.is_available()
    
    # Hyperparameter set
    configs.setting = '{}_{}_{}'.format(
                    configs.learning_rate,
                    configs.batch_size,
                    configs.moving_avg)

    logger.info('Loading the datasets...')
    dataset_all = Dataset_all(configs)
    cuda_exist = torch.cuda.is_available()
    torch.cuda.init()

    ################################## MODEL INIT
    if cuda_exist:
        configs.device = torch.device('cuda')
        logger.info('Using Cuda...')
        model = Net(configs).cuda()

    else:
        configs.device  = torch.device('cpu')
        logger.info('Not using cuda...')
        model = Net(configs)
    ################################## MODEL INIT
    
    logger.info(configs)
    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    logger.info('Starting training for {} epoch(s)'.format(configs.train_epochs))
    train_and_evaluate(model,
                    dataset_all,
                    optimizer,
                    configs)
                            