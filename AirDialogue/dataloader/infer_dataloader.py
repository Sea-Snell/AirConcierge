import dataloader as normal_dataloader
import selfplay_dataloader as selfplay_dataloader
# import utils.simulate_DB_module as simulate
from torch.utils.data import DataLoader
from functools import partial
import os

UNK = "<unk>"

def get_prior_items(batch_size, toy, max_len=None, need_shuffle=False, mask=False, only_f=False, dev=False, n_sample=-1, small_n=30, args=None):
    global eod_id
    global smallkb_n
    smallkb_n = small_n
    normal_dataloader.smallkb_n = smallkb_n
    # print('Kb_n : ', smallkb_n)

    if args.syn:
        pre_data_path = './data/synthesized/'
    elif args.air:
        pre_data_path = './data/airdialogue/'
    else:
        print('Pleae use --syn or --air !')
        raise

    # test
    vocab_file = pre_data_path + 'tokenized/vocab.txt'
    if toy: 
        src_data_file = pre_data_path + 'tokenized/sub_air/toy_dev.selfplay.eval.data'
        kb_file = pre_data_path + 'tokenized/sub_air/toy_dev.selfplay.eval.kb'
        table_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.tables.jsonl'
        sql_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.jsonl'
    else:
        src_data_file = pre_data_path + 'tokenized/selfplay_eval/dev.selfplay.eval.data'
        kb_file = pre_data_path + 'tokenized/selfplay_eval/dev.selfplay.eval.kb'
        table_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.tables.jsonl'
        sql_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.jsonl'

    print('SelfPlayEval_loader Loading data : ', src_data_file)
    print('SelfPlayEval_loader Loading kb : ', kb_file)

    # vocab table & tokenize
    corpus = normal_dataloader.Corpus()
    corpus.create_vocab_table(vocab_file) # self.dictionary.add_word(word)
    sents, kb_true_answer, action_state, SQL_YN = corpus.self_play_eval_tokenize(src_data_file)
    kb_sents = corpus.tokenize_kb(kb_file)
    sql_data, table_data = corpus.tokenize_column(sql_path, table_path)

    if n_sample != -1:
        sents = sents[:n_sample]
        kb_sents = kb_sents[:n_sample]
        SQL_YN = SQL_YN[:n_sample]
        kb_true_answer = kb_true_answer[:n_sample]
        sql_data = sql_data[:n_sample]

    combined_dataset = list(sents)
    for i in range(len(sents)):
        combined_dataset[i].append(kb_sents[i])
        combined_dataset[i].append(kb_true_answer[i])
        combined_dataset[i].append(SQL_YN[i])
        combined_dataset[i].append(sql_data[i])

    if only_f:
        combined_dataset_new = []
        for i in range(len(sents)):
            if action_state[i] != -1 and SQL_YN[i] == 1:
                combined_dataset_new.append(combined_dataset[i])
        print('combined_dataset new : ', len(combined_dataset_new), 100. * len(combined_dataset_new) / len(combined_dataset))
        combined_dataset = combined_dataset_new

    eod_id = corpus.dictionary.word2idx['<eod>']
    normal_dataloader.eod_id = eod_id
    t1_id = corpus.dictionary.word2idx['<t1>']
    t2_id = corpus.dictionary.word2idx['<t2>']
    unk = corpus.dictionary.word2idx['<unk>']
    combined_dataset = map(partial(normal_dataloader.process_entry_self_play_eval, vocab_table=corpus, t1_id=t1_id, t2_id=t2_id, table_data=table_data), combined_dataset)
    return combined_dataset, corpus

def get_selfplay_items(batch_size, toy, max_len=None, need_shuffle=False, mask=False, only_f=False, dev=False, n_sample=-1, small_n=30, args=None):
    global eod_id
    global smallkb_n
    smallkb_n = small_n
    selfplay_dataloader.smallkb_n = smallkb_n
    # print('Kb_n : ', smallkb_n)

    if args.syn:
        pre_sql_path = './results/synthesized/'
        pre_data_path = './data/synthesized/'
    elif args.air:
        pre_sql_path = './results/airdialogue/'
        pre_data_path = './data/airdialogue/'
    else:
        print('Pleae use --syn or --air !')
        raise

    # test
    vocab_file = pre_data_path + 'tokenized/vocab.txt'
    if toy :
        src_data_file = pre_data_path + 'tokenized/sub_air/toy_dev.selfplay.eval.data'
        kb_file = pre_sql_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered.kb'
        table_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.tables.jsonl'
        sql_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.jsonl'
        filtered_index_path = pre_sql_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered_index.kb'
        # turn_gate_path = pre_sql_path + 'SelfPlay_Eval/SQL/prior_gate.txt'
    else:
        src_data_file = pre_data_path + 'tokenized/selfplay_eval/dev.selfplay.eval.data'
        kb_file = pre_sql_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered.kb'
        table_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.tables.jsonl'
        sql_path = pre_data_path + 'SQL/dev_selfplay_eval/selfplay_eval_tok.jsonl'
        filtered_index_path = pre_sql_path + 'SelfPlay_Eval/SQL/dev_sql/simulate_DB/filtered_index.kb'
        # turn_gate_path = pre_sql_path + 'SelfPlay_Eval/SQL/prior_gate.txt'

    print('SelfPlayEval_loader Loading data : ', src_data_file)
    print('SelfPlayEval_loader Loading kb : ', kb_file)

    # vocab table & tokenize
    corpus = selfplay_dataloader.Corpus()
    corpus.create_vocab_table(vocab_file) # self.dictionary.add_word(word)
    sents, kb_true_answer, action_state, SQL_YN = corpus.self_play_eval_tokenize(src_data_file)
    kb_sents = corpus.tokenize_kb(kb_file)
    sql_data, table_data = corpus.tokenize_column(sql_path, table_path)
    filter_index = selfplay_dataloader.read_fileter_kb(filtered_index_path)
    # turn_gate = selfplay_dataloader.read_turn_gate(turn_gate_path)

    combined_dataset = list(sents)
    for i in range(len(sents)):
        combined_dataset[i].append(kb_sents[i])
        combined_dataset[i].append(kb_true_answer[i])
        combined_dataset[i].append(SQL_YN[i])
        combined_dataset[i].append(sql_data[i])
        combined_dataset[i].append(filter_index[i])
        combined_dataset[i].append(0)

    if only_f:
        combined_dataset_new = []
        for i in range(len(sents)):
            if action_state[i] != -1 and SQL_YN[i] == 1:
                combined_dataset_new.append(combined_dataset[i])
        print('combined_dataset new : ', len(combined_dataset_new), 100. * len(combined_dataset_new) / len(combined_dataset))
        combined_dataset = combined_dataset_new

    eod_id = corpus.dictionary.word2idx['<eod>']
    selfplay_dataloader.eod_id = eod_id
    t1_id = corpus.dictionary.word2idx['<t1>']
    t2_id = corpus.dictionary.word2idx['<t2>']
    unk = corpus.dictionary.word2idx['<unk>']
    combined_dataset = map(partial(selfplay_dataloader.process_entry_self_play_eval, vocab_table=corpus, t1_id=t1_id, t2_id=t2_id, table_data=table_data), combined_dataset)
    return combined_dataset, corpus

def basic_tokenize(dictionary, utterances, intent_goal, final_state, mask=False, add_end=False):
    items = []
    for speaker, utterance in utterances:
        utterance = utterance.replace('.', ' . ').replace('?', ' ? ').replace('!', ' ! ').replace(',', ' , ').strip()
        if speaker == 'Customer':
            items.append('<t1>')
        elif speaker == 'Agent':
            items.append('<t2>')
        else:
            raise NotImplementedError
        items.append(utterance)
    if add_end:
        items.append('<eod>')
    if utterances[-1][0] == 'Customer':
        items.append('<t2>')
    elif utterances[-1][0] == 'Agent':
        items.append('<t1>')
    else:
        raise NotImplementedError
    combined_tokens = ' '.join(items)
    words = []
    for word in combined_tokens.split(" "):
        word = word.strip()
        if len(word) == 0:
            continue
        if mask and ((intent_goal == 'book' and final_state == 'book') or (intent_goal == 'change' and final_state == 'change')):
            if word.isdigit() and (int(word) >= 1001 and int(word) <= 1029):
                # word = '<mask_flight>'
                word = '<fl_' + str(word) + '>'
            else:
                for f in range(1001, 1030):
                    str_f = str(f)
                    if str_f in word:
                        # word = '<mask_flight>'
                        word = '<fl_' + str_f + '>'
            if '1000' in word and ('ight' in words[-4:] or 'umber' in words[-4:]):
                # word = '<mask_flight>'
                word = '<fl_1000>'
        try:
            words.append(dictionary.word2idx[word])
        except KeyError:
            words.append(dictionary.word2idx[UNK])
    return words

def combined_collate_fn(items, include_kb=False):
    prior_items, selfplay_items = zip(*items)
    return (normal_dataloader.collate_fn_self_play_eval(prior_items, include_kb=include_kb), 
            selfplay_dataloader.collate_fn_self_play_eval(selfplay_items, include_kb=include_kb),)

def Infer_loader(batch_size, toy, max_len=None, need_shuffle=False, mask=False, only_f=False, dev=False, n_sample=-1, small_n=30, args=None, include_kb=False):
    prior_items, prior_corpus = get_prior_items(batch_size, toy, max_len=max_len, 
                                                need_shuffle=need_shuffle, mask=mask,
                                                only_f=only_f, dev=dev, n_sample=n_sample, 
                                                small_n=small_n, args=args)
    selfplay_items, selfplay_corpus = get_selfplay_items(batch_size, toy, max_len=max_len, 
                                                         need_shuffle=need_shuffle, mask=mask, only_f=only_f, 
                                                         dev=dev, n_sample=n_sample, small_n=small_n, args=args)

    data = normal_dataloader.AirData(zip(prior_items, selfplay_items))
    if dev :
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=partial(combined_collate_fn, include_kb=include_kb), drop_last=False)

    return data_loader, prior_corpus, selfplay_corpus