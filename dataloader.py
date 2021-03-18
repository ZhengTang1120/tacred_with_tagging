import json
import constant
import string

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, intervals, patterns, tokenizer, evaluation=False):
        self.filename = filename
        self.intervals = intervals
        self.patterns = patterns
        self.tokenizer = tokenizer 
        self.label2id = constant.LABEL_TO_ID

        data = self.preprocess()
        return data
    
    def preprocess(self):
        processed = []
        with open(self.intervals) as f:
            intervals = f.readlines()
        with open(self.patterns) as f:
            patterns = f.readlines()
        for c, d in enumerate(json.load(open(self.filename))):
            tokens = list(d['token'])
            tokens = [t.lower() for t in tokens]
            ner = d['stanford_ner']
            relation = self.label2id[d['relation']]

            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['[SUBJ-'+d['subj_type']+']'] * (se-ss+1)
            tokens[os:oe+1] = ['[OBJ-'+d['obj_type']+']'] * (oe-os+1)
            rl, masked = intervals[c].split('\t')
            rl, pattern = patterns[c].split('\t')
            masked = eval(masked)
            if masked and d['relation'] != 'no_relation':
                masked = list(range(masked[0], masked[1]))
                for i in range(len(masked)):
                    if masked[i] < min(os, ss):
                        masked[i] += 1
                    elif masked[i] <= min(se,oe):
                        masked[i] += 2
                    elif masked[i] < max(os, ss):
                        masked[i] += 3
                    elif masked[i] <= max(se, oe):
                        masked[i] += 4
                    else:
                        masked[i] += 5
                has_tag = True
            else:
                pattern = ''
                masked = []
                has_tag = False
            if ss<os:
                os = os + 2
                oe = oe + 2
                tokens.insert(ss, '#')
                tokens.insert(se+2, '#')
                tokens.insert(os, '$')
                tokens.insert(oe+2, '$')
                ner.insert(ss, '#')
                ner.insert(se+2, '#')
                ner.insert(os, '$')
                ner.insert(oe+2, '$')
            else:
                ss = ss + 2
                se = se + 2
                tokens.insert(os, '$')
                tokens.insert(oe+2, '$')
                tokens.insert(ss, '#')
                tokens.insert(se+2, '#')
                ner.insert(os, '$')
                ner.insert(oe+2, '$')
                ner.insert(ss, '#')
                ner.insert(se+2, '#')
            tokens = ['[CLS]'] + tokens
            ner = ['[CLS]'] + ner
            l = len(tokens)
            subj_positions = get_positions(ss+2, se+2, l)
            obj_positions = get_positions(os+2, oe+2, l)
            if has_tag and relation!=0:
                tagging = [0 if i not in masked else 1 if (tokens[i] in pattern or ner[i] in pattern) and tokens[i] not in string.punctuation else 0 for i in range(len(tokens))]
            # elif relation!=0:
            #     tagging = [1 if i !=0 else 0 for i in range(len(tokens))]
            else:
                tagging = [0 for i in range(len(tokens))]
            if has_tag:
                print (pattern)
                print ([(tokens[i], ner[i], tagging[i], subj_positions[i], obj_positions[i]) for i in range(l)])
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            processed += [(tokens, subj_positions, obj_positions, relation, tagging, has_tag)]
        return processed

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return [0] * start_idx + [1]*(end_idx - start_idx + 1) + [0] * (length-end_idx-1)

