from transformers import BertTokenizer

from dataloader import DataLoader
import constant

tokenizer = BertTokenizer.from_pretrained('mrm8488/spanbert-large-finetuned-tacred')
special_tokens_dict = {'additional_special_tokens': constant.ENTITY_TOKENS}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

train = DataLoader(opt['data_dir'] + '/train.json', opt['data_dir'] + '/interval_train.txt', opt['data_dir'] + '/pattern_train.txt', tokenizer)