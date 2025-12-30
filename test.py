#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

from types import SimpleNamespace
from train import *
# In[2]:

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

MODEL_FILENAME_DICT = {
    "DLinear": DLinear,
    }

logger = logging.getLogger('best_hyperparameter.run')
parser = argparse.ArgumentParser()

parser.add_argument('--data', type = str, default='brw', help = 'dataset type')
parser.add_argument('--model_type', type = str, default='DLinear', help='model type')
parser.add_argument('--model_name', type = str, default='model', help='Directory containing params.json')
parser.add_argument('--loss', type = str, default='CRPS', help='Directory containing params.json')
parser.add_argument('--pred_len', type = int, default=24, help='prediction length')
parser.add_argument('--random_seed', type = int, default=2025, help='Random Seed num')


if __name__ == '__main__':

    configs = parser.parse_args()
    set_seed(configs.random_seed)
    configs.model_dir = os.path.join(configs.model_name, configs.model_type, str(configs.pred_len), configs.data)
    
    cuda_exist = torch.cuda.is_available()

    ## Hyperparameter setting
    learning_rate_list = [0.0001, 0.001]
    batch_size_list = [32, 64]
    moving_avg_list = [7, 25, 73]


    best_test_score = -np.inf
    val_dict = {}
    test_dict = {}
    for learning_rate, batch_size, moving_avg in itertools.product(learning_rate_list,batch_size_list,moving_avg_list) :
        configs.learning_rate = learning_rate
        configs.batch_size = batch_size
        configs.moving_avg = moving_avg
        configs.setting = '{}_{}_{}'.format(
                        configs.learning_rate,
                        configs.batch_size,
                        configs.moving_avg)

        
        # Validation
        path = os.path.join(configs.model_dir, configs.setting)

        json_path =  path + '/' + 'configs.json'
        configs_dict = load_json(json_path)
        best_score = configs_dict['best score']


        if best_score > best_test_score :
            best_test_score = best_score
            best_json_path = os.path.join(configs.model_dir,  f'best_configs.json')
            save_dict_to_json(configs_dict, best_json_path)
    

    # Best model and predict testset
    best_configs = load_json(best_json_path)
    best_model_path = os.path.join(best_configs['model_dir'],best_configs['setting'])
    best_model_path = best_model_path + '/' + 'checkpoint.pth'
    best_configs = SimpleNamespace(**best_configs)
    model = MODEL_FILENAME_DICT[configs.model_type](best_configs).cuda()
    checkpoint = torch.load(best_model_path)  # Load dict from file
    model.load_state_dict(checkpoint)         # Pass dict to load_state_dict

    # Load test set
    dataset_all = Dataset_all(best_configs)
    test_set, test_loader = dataset_all._get_data('test')
    summary_test = evaluate(model, test_set, test_loader, best_configs, loss_type =["CRPS", "QLOSS", "ECRPS", "QCL", "MSE", "MAE"])

    results_path = os.path.join(f'./result/{str(configs.pred_len)}/' + best_configs.model_type + 'NQF_' + best_configs.data)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    output_path =  os.path.join(results_path, 'results.json')
    save_dict_to_json(summary_test, output_path)