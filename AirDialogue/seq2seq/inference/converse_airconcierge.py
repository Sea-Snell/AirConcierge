from __future__ import division
import logging
import os
import random
import time

import torch
# import torchtext
from torch import optim

import seq2seq
from seq2seq.loss import *
from seq2seq.query import *
from seq2seq.database import *
from seq2seq.util.checkpoint import Checkpoint
from utils.utils import *
from tensorboardX import SummaryWriter 
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence
from dataloader.infer_dataloader import basic_tokenize
# from utils import simulate_DB

class ConverseModel(object):
    def __init__(self, model_dir='checkpoints/', args=None, prior_corpus=None, selfplay_corpus=None):
        self._trainer = "Simple Inference"

        if not os.path.exists(model_dir):
            print('No such model dir !')
            raise
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.prior_corpus = prior_corpus
        self.selfplay_corpus = selfplay_corpus

    def list_to_tensor(self, ll):
        tensor_list = []
        for i in ll:
            tensor_list.append(torch.tensor(np.array(i).astype(int)))
        return tensor_list

    def check_boundary(self, dialogue):
        t1_list = []
        t2_list = []
        index = 0
        for token in dialogue:
            if token == '<t1>':
                t1_list.append(str(index))
            if token == '<t2>':
                t2_list.append(str(index))
            if token == '<eod>':
                break
            index = index + 1
        return t1_list + t2_list

    def translate_query_to_simple(self, query):
        condiction = ['departure_airport', 'return_airport', 'departure_month', 'return_month', 'departure_day', 'return_day', 'departure_time_num', 'return_time_num', 'class', \
                     'price', 'num_connections', 'airline_preference']

        simple_query = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        curr = 0
        for i in range(12):
            if curr == len(query):
                break
            if condiction[i] in query[curr]:
                simple_query[i] = int(query[curr].split()[-1])
                curr += 1
        return simple_query
    
    def converse_to_gate(self, dialogue_tokens_lists, model, intent, size_intent, action, has_reservation, col_seq, truth_seq, SQL_YN, kb_true_answer, args=None):
        with torch.no_grad():
            model.eval()
            intent, size_intent, action, has_reservation, col_seq, SQL_YN, kb_true_answer = intent.cuda(), size_intent.cuda(), action.cuda(), has_reservation.cuda(), col_seq.cuda(), SQL_YN.cuda(), kb_true_answer.cuda()

            b = intent.size(0)
            intent_sents = [[] for _ in range(b)]
            action_sents = [[] for _ in range(b)]
            action_sentsw = [[] for _ in range(b)]
            for i in range(b):
                for j in intent[i]:
                    intent_sents[i].append(self.prior_corpus.dictionary.idx2word[j.data])
                for j in range(2):
                    action_sents[i].append(self.prior_corpus.dictionary.idx2word[action[i,j].data])
                    action_sentsw[i].append(self.prior_corpus.dictionary.idx2word[action[i,j].data])
                action_sents[i].append('<fl_'+str(int(action[i,2].data) + 1000)+'>')
                if int(action[i,2].data) == 30:
                    action_sentsw[i].append('<fl_empty>')
                else:
                    action_sentsw[i].append('<fl_'+str(int(action[i,2].data) + 1000)+'>')
                action_sents[i].append(self.prior_corpus.dictionary.idx2word[action[i,3].data])
                action_sentsw[i].append(self.prior_corpus.dictionary.idx2word[action[i,3].data])

            history = [[1]] # <t1>
            history_sents = [['<t1>']]
            request = 0
            eod_id= self.prior_corpus.dictionary.word2idx['<eod>']
            end = 0
            t2_gate_turn = -1

            for turn_i in range(args.max_dialogue_turns):
                turn, dialogue_tokens = dialogue_tokens_lists[turn_i]
                if turn == 'Customer':
                    history = [dialogue_tokens] # <t1>
                    history_sents = [list(map(lambda x: self.prior_corpus.dictionary.idx2word[x], dialogue_tokens))]

                    if args.sql:
                        # list to tensor
                        source_diag = pad_sequence(self.list_to_tensor(history), batch_first=True, padding_value=eod_id); source_diag = source_diag.cuda()
                        b = source_diag.size(0)
                        size_history = []
                        for i in range(b):
                            size_history.append(len(history[i]))
                        size_dialogue = torch.tensor(size_history, dtype=torch.int64); size_dialogue = size_dialogue.cuda()
                        # display source input 
                        source_diag_sents = [[] for _ in range(b)]
                        for i in range(b):
                            for j in source_diag[i]:
                                source_diag_sents[i].append(self.prior_corpus.dictionary.idx2word[j.data])
                        logits_train1, sequence_symbols = model.module.Call_t1_SelfPlayEval(intent, size_intent, source_diag, size_dialogue, args=args)
                        
                    # ('sequence_symbols : ', sequence_symbols)
                    sents = [[] for _ in range(b)]
                    length = [0 for _ in range(b)]
                    for s in range(len(sequence_symbols)):
                        for i in range(b):
                            token = self.prior_corpus.dictionary.idx2word[sequence_symbols[s][i]]
                            if token == '<t2>' and length[i] == 0: # end of sentence
                                sents[i].append(token)
                                length[i] = len(sents[i])
                                history[i].append(sequence_symbols[s][i])
                                history_sents[i].append(token)
                            elif token != '<t2>' and length[i] == 0:
                                sents[i].append(token)
                                history[i].append(sequence_symbols[s][i])
                                history_sents[i].append(token)
                    for i in range(b):
                        if '<eod>' in sents[i]:
                            end = 1
                            boundary_list = self.check_boundary(history_sents[i])
                            break
                elif turn == 'Agent':
                    history = [dialogue_tokens] # <t1>
                    history_sents = [list(map(lambda x: self.prior_corpus.dictionary.idx2word[x], dialogue_tokens))]

                    if args.sql:
                        # Dialogue input : list to tensor
                        source_diag = pad_sequence(self.list_to_tensor(history), batch_first=True, padding_value=eod_id); source_diag = source_diag.cuda()
                        b = source_diag.size(0)
                        size_history = []
                        for i in range(b):
                            size_history.append(len(history[i]))
                        size_dialogue = torch.tensor(size_history, dtype=torch.int64); size_dialogue = size_dialogue.cuda()
                        # Display source input 
                        source_diag_sents = [[] for _ in range(b)]
                        for i in range(b):
                            for j in source_diag[i]:
                                source_diag_sents[i].append(self.prior_corpus.dictionary.idx2word[j.data])
                        
                        # Send query ? + Generate SQL 
                        cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate = model.module.SQL_AND_StateTracking(source_diag, size_dialogue, col_seq, args=args)
                        request = predicted_gate[0][-1].data.item()
                        request_index = predicted_gate[0].size(0) - 1
                        if request == 1.:
                            SQL_query, pred, pred_gate = Generate_SQL(cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate, truth_seq, SQL_YN, (None, None, None,), args=args)
                            t2_gate_turn = turn_i
                            end = 1
                            break
                        else:
                            concat_flight = torch.zeros((1, 1, 256)); concat_flight = concat_flight.cuda()
                            logits_train2, sequence_symbols, predicted_gate = model.module.Call_t2_SelfPlayEvalPrior(source_diag, size_dialogue, has_reservation, col_seq, concat_flight, args=args)

                            ################################################################
                            ######################### Prior ################################
                            ################################################################
                            prior_history = [list(history[0])]
                            prior_sents = [[] for _ in range(b)]
                            prior_length = [0 for _ in range(b)]
                            for s in range(len(sequence_symbols)):
                                for i in range(b):
                                    token = self.prior_corpus.dictionary.idx2word[sequence_symbols[s][i]]
                                    if token == '<t1>' and prior_length[i] == 0:
                                        prior_sents[i].append(token)
                                        prior_length[i] = len(sents[i])
                                        prior_history[i].append(sequence_symbols[s][i])
                                    elif token != '<t1>' and prior_length[i] == 0:
                                        prior_sents[i].append(token)
                                        prior_history[i].append(sequence_symbols[s][i])
                            
                            # list to tensor
                            prior_source_diag = pad_sequence(self.list_to_tensor(prior_history), batch_first=True, padding_value=eod_id); prior_source_diag = prior_source_diag.cuda()
                            b = prior_source_diag.size(0)
                            prior_size_history = []
                            for i in range(b):
                                prior_size_history.append(len(prior_history[i]))
                            prior_size_dialogue = torch.tensor(prior_size_history, dtype=torch.int64); prior_size_dialogue = prior_size_dialogue.cuda()
                            prior_predicted_gate = model.module.Prior_Gate(prior_source_diag, prior_size_dialogue, col_seq)

                            if torch.sum(prior_predicted_gate[i]).data.item() != 0 and request == 0:
                                fix_predict_gate = predicted_gate
                                fix_predict_gate[0][request_index] = prior_predicted_gate[0][request_index+1]
                                predicted_gate = fix_predict_gate

                                request = 1.
                                request_index = predicted_gate[0].size(0) - 1
                                SQL_query, pred, pred_gate = Generate_SQL(cond_op_score, cond_col_score, cond_num_score, cond_str_out, predicted_gate, truth_seq, SQL_YN, (None, None, None,), args=args)
                                t2_gate_turn = turn_i
                                end = 1

                            ################################################################
                            ######################### Prior ################################
                            ################################################################

                    # ('sequence_symbols : ', sequence_symbols)
                    sents = [[] for _ in range(b)]
                    length = [0 for _ in range(b)]
                    for s in range(len(sequence_symbols)):
                        for i in range(b):
                            token = self.prior_corpus.dictionary.idx2word[sequence_symbols[s][i]]
                            if token == '<t1>' and length[i] == 0:
                                sents[i].append(token)
                                length[i] = len(sents[i])
                                history[i].append(sequence_symbols[s][i])
                                history_sents[i].append(token)
                            elif token != '<t1>' and length[i] == 0:
                                sents[i].append(token)
                                history[i].append(sequence_symbols[s][i])
                                history_sents[i].append(token)
                    
                    for i in range(b):
                        if '<eod>' in sents[i]:
                            end = 1
                            boundary_list = self.check_boundary(history_sents[i])
                            history_gate = []
                            for h in range(len(history_sents[i])):
                                if h < predicted_gate[i].size(0):
                                    history_gate.append(str(history_sents[i][h]) + '_' + str(predicted_gate[i][h].data.item()))
                                else:
                                    history_gate.append(history_sents[i][h])
                            break
                if end == 1 or turn_i >= (len(dialogue_tokens_lists) - 1):
                    break
        return t2_gate_turn, history[0]
    
    def converse_from_gate(self, dialogue_tokens_lists, model, intent, size_intent, action, kb, has_reservation, col_seq, truth_seq, SQL_YN, turn_gate, args=None):
        turn_gate = [turn_gate]
        with torch.no_grad(): 
            intent, size_intent, action, kb, has_reservation, col_seq, SQL_YN = intent.cuda(), size_intent.cuda(), action.cuda(), kb.cuda(), has_reservation.cuda(), col_seq.cuda(), SQL_YN.cuda()
            
            # intent and action 
            b = intent.size(0)
            intent_sents = [[] for _ in range(b)]
            action_sents = [[] for _ in range(b)]
            action_sentsw = [[] for _ in range(b)]
            for i in range(b):
                for j in intent[i]:
                    intent_sents[i].append(self.selfplay_corpus.dictionary.idx2word[j.data])
                for j in range(2):
                    action_sents[i].append(self.selfplay_corpus.dictionary.idx2word[action[i,j].data])
                    action_sentsw[i].append(self.selfplay_corpus.dictionary.idx2word[action[i,j].data])
                action_sents[i].append('<fl_'+str(int(action[i,2].data) + 1000)+'>')
                if int(action[i,2].data) == 30:
                    action_sentsw[i].append('<fl_empty>')
                else:
                    action_sentsw[i].append('<fl_'+str(int(action[i,2].data) + 1000)+'>')
                action_sents[i].append(self.selfplay_corpus.dictionary.idx2word[action[i,3].data])
                action_sentsw[i].append(self.selfplay_corpus.dictionary.idx2word[action[i,3].data])
                
            history = [[1]] # <t1>
            history_sents = [['<t1>']]
            request = 0
            eod_id= self.selfplay_corpus.dictionary.word2idx['<eod>']
            end = 0
            concat_flight_embed = torch.zeros((1, 1, 256)); concat_flight_embed = concat_flight_embed.cuda()
            turn_index = -1
            predict_flight_number = 'None'

            for turn_i in range(args.max_dialogue_turns):
                turn, dialogue_tokens = dialogue_tokens_lists[turn_i]
                if turn == 'Customer':
                    history = [dialogue_tokens] # <t1>
                    history_sents = [list(map(lambda x: self.prior_corpus.dictionary.idx2word[x], dialogue_tokens))]

                    if args.sql:
                        # list to tensor
                        source_diag = pad_sequence(self.list_to_tensor(history), batch_first=True, padding_value=eod_id); source_diag = source_diag.cuda()
                        b = source_diag.size(0)
                        size_history = []
                        for i in range(b):
                            size_history.append(len(history[i]))
                        size_dialogue = torch.tensor(size_history, dtype=torch.int64); size_dialogue = size_dialogue.cuda()
                        # display source input 
                        source_diag_sents = [[] for _ in range(b)]
                        for i in range(b):
                            for j in source_diag[i]:
                                source_diag_sents[i].append(self.selfplay_corpus.dictionary.idx2word[j.data])
                        logits_train1, sequence_symbols = model.module.Call_t1_SelfPlayEval(intent, size_intent, source_diag, size_dialogue, args=args)
                        
                    # ('sequence_symbols : ', sequence_symbols)
                    sents = [[] for _ in range(b)]
                    length = [0 for _ in range(b)]
                    for s in range(len(sequence_symbols)):
                        for i in range(b):
                            token = self.selfplay_corpus.dictionary.idx2word[sequence_symbols[s][i]]
                            if token == '<t2>' and length[i] == 0:
                                sents[i].append(token)
                                length[i] = len(sents[i])
                                history[i].append(sequence_symbols[s][i])
                                history_sents[i].append(token)
                            elif token != '<t2>' and length[i] == 0:
                                sents[i].append(token)
                                history[i].append(sequence_symbols[s][i])
                                history_sents[i].append(token)
                    for i in range(b):
                        if '<eod>' in sents[i]:
                            end = 1
                            break
                if turn == 'Agent':
                    history = [dialogue_tokens] # <t1>
                    history_sents = [list(map(lambda x: self.prior_corpus.dictionary.idx2word[x], dialogue_tokens))]

                    if args.sql:
                        # Dialogue input : list to tensor
                        source_diag = pad_sequence(self.list_to_tensor(history), batch_first=True, padding_value=eod_id); source_diag = source_diag.cuda()
                        b = source_diag.size(0)
                        size_history = []
                        for i in range(b):
                            size_history.append(len(history[i]))
                        size_dialogue = torch.tensor(size_history, dtype=torch.int64); size_dialogue = size_dialogue.cuda()
                        # Display source input 
                        source_diag_sents = [[] for _ in range(b)]
                        for i in range(b):
                            for j in source_diag[i]:
                                source_diag_sents[i].append(self.selfplay_corpus.dictionary.idx2word[j.data])

                        # Send query ? + Generate SQL 
                        if turn_i == turn_gate[0]:
                            request = 1
                            # print('initi kb : ', kb.size())
                            if kb[0].size(0) == 1 : # empty
                                # print('kb : ', kb[0].size())
                                # print('Empty')
                                pass
                            elif kb[0].size(0) == 2 : # one flight
                                # print('kb : ', kb[0].size())
                                # print('One flight')
                                concat_flight = kb[0, 0:1]
                                # print('concat_flight : ', concat_flight.size())
                                predict_flight_number = self.selfplay_corpus.dictionary.idx2word[concat_flight[0, -1]]
                                # print('predict_flight_number : ', predict_flight_number)
                                concat_flight_embed = model.module.Encode_Flight_KB(concat_flight.unsqueeze(0))
                            else: # result > 1
                                # print('More than one flight')
                                global_pointer = model.module.Point_Encode_KB(source_diag, size_dialogue, kb, has_reservation, col_seq, args=args)
                                # print('global_pointer : ', global_pointer)
                                _, global_pointer_index = torch.max(F.softmax(global_pointer, dim=1).data, 1)
                                # print('global_pointer_index : ', global_pointer_index)
                                concat_flight = kb[0, global_pointer_index]
                                # print('concat_flight : ', concat_flight.size(), concat_flight)
                                predict_flight_number = self.selfplay_corpus.dictionary.idx2word[concat_flight[0, -1]]
                                # print('predict_flight_number : ', predict_flight_number)
                                concat_flight_embed = model.module.Encode_Flight_KB(concat_flight.unsqueeze(0))
                            logits_train2, sequence_symbols, predicted_gate = model.module.Call_t2_SelfPlayEval_2(source_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, args=args)
                            turn_index = predicted_gate[0].size(0) - 1
                        
                        if request == 1:
                            logits_train2, sequence_symbols, predicted_gate = model.module.Call_t2_SelfPlayEval_2(source_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, turn_gate=turn_index, args=args)
                        else:
                            logits_train2, sequence_symbols, predicted_gate = model.module.Call_t2_SelfPlayEval_2(source_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, args=args)
                    
                    # ('sequence_symbols : ', sequence_symbols)
                    sents = [[] for _ in range(b)]
                    length = [0 for _ in range(b)]
                    for s in range(len(sequence_symbols)):
                        for i in range(b):
                            token = self.selfplay_corpus.dictionary.idx2word[sequence_symbols[s][i]]
                            if token == '<t1>' and length[i] == 0:
                                sents[i].append(token)
                                length[i] = len(sents[i])
                                history[i].append(sequence_symbols[s][i])
                                history_sents[i].append(token)
                            elif token != '<t1>' and length[i] == 0:
                                sents[i].append(token)
                                history[i].append(sequence_symbols[s][i])
                                history_sents[i].append(token)
                    if args.air:
                        if turn_i == turn_gate[0]:
                            add_token = ['flight', 'number', 'is', predict_flight_number, '.', '<t1>']
                            if any('<fl_' in s for s in sents[0]) == False and '<fl_10' in predict_flight_number:
                                history[i] = history[i][:-1] # pop out <t1>
                                history_sents[i] = history_sents[i][:-1] # pop out <t1>
                                for add_s in add_token:
                                    history[i].append(self.selfplay_corpus.dictionary.word2idx[add_s])
                                    history_sents[i].append(add_s)

                    for i in range(b):
                        if '<eod>' in sents[i]:
                            end = 1
                            break
                if end == 1 or turn_i >= (len(dialogue_tokens_lists)-1):
                    break
        
            if end == 1:
                source_diag = pad_sequence(self.list_to_tensor(history), batch_first=True, padding_value=eod_id); source_diag = source_diag.cuda()
                b = source_diag.size(0)
                size_history = []
                for i in range(b):
                    size_history.append(len(history[i]))
                size_dialogue = torch.tensor(size_history, dtype=torch.int64); size_dialogue = size_dialogue.cuda()
                # display source input 
                source_diag_sents = [[] for _ in range(b)]
                for i in range(b):
                    for j in source_diag[i]:
                        source_diag_sents[i].append(self.selfplay_corpus.dictionary.idx2word[j.data])
                logits_train3 = model.module.Call_t2_SelfPlayEval_2(source_diag, size_dialogue, has_reservation, col_seq, concat_flight_embed, turn_gate=turn_index, end=1, args=args)
                _, predict_action_temp = compute_action_nn(logits_train3, action)
                predict_action_name1 = self.selfplay_corpus.dictionary.idx2word[predict_action_temp[0][0].data.item()]
                predict_action_name2 = self.selfplay_corpus.dictionary.idx2word[predict_action_temp[1][0].data.item()]
                predict_action_flight = predict_action_temp[2][0].data.item()
                if predict_action_flight == 30:
                    predict_action_flight = 0
                else:
                    predict_action_flight = int(predict_action_flight)+1000
                predict_action_state = self.selfplay_corpus.dictionary.idx2word[predict_action_temp[3][0].data.item()]
                predict_action = (predict_action_state.split('_', 1)[1].split('>', 1)[0], predict_action_name1 + ' ' + predict_action_name2, predict_action_flight)
            else:
                predict_action = None
        
        return history[0], predict_action

    def converse_one(self, utterances, model, prior_items, selfplay_items, args=None):
        intent, size_intent, action, has_reservation, col_seq, truth_seq, SQL_YN, kb_true_answer = prior_items
        intent_strs = list(map(lambda x: self.prior_corpus.dictionary.idx2word[x], intent[0]))
        final_state_strs = list(map(lambda x: self.prior_corpus.dictionary.idx2word[x], action[0]))
        intent_goal = intent_strs[14].split('_', 1)[1].split('>', 1)[0]
        if len(final_state_strs) != 4:
            final_state = final_state_strs[1].split('_', 1)[1].split('>', 1)[0]
        else:
            final_state = final_state_strs[3].split('_', 1)[1].split('>', 1)[0]
        dialogue_tokens_lists_prior = [(utterances[0][0], [self.prior_corpus.dictionary.word2idx['<t1>' if utterances[0][0] == 'Customer' else '<t2>']])]
        dialogue_tokens_lists_selfplay = [(utterances[0][0], [self.selfplay_corpus.dictionary.word2idx['<t1>' if utterances[0][0] == 'Customer' else '<t2>']])]
        for i, (speaker, utterance) in enumerate(utterances):
            if speaker == 'Agent' or speaker == 'Customer':
                dialogue_tokens_lists_prior.append(('Agent' if speaker == 'Customer' else 'Customer', basic_tokenize(self.prior_corpus.dictionary, utterances[:(i+1)], intent_goal, final_state, mask=True, add_end=False)))
                dialogue_tokens_lists_selfplay.append(('Agent' if speaker == 'Customer' else 'Customer', basic_tokenize(self.selfplay_corpus.dictionary, utterances[:(i+1)], intent_goal, final_state, mask=True, add_end=False)))
            else:
                raise NotImplementedError

        turn_gate, new_history = self.converse_to_gate(dialogue_tokens_lists_prior, model, intent, size_intent, action, has_reservation, 
                                       col_seq, truth_seq, SQL_YN, kb_true_answer, args=args)
        new_str = ' '.join(map(lambda x: self.prior_corpus.dictionary.idx2word[x], new_history))
        predict_action = None
        intent, size_intent, action, kb, has_reservation, col_seq, truth_seq, SQL_YN, _ = selfplay_items
        new_history, predict_action = self.converse_from_gate(dialogue_tokens_lists_selfplay, model, intent, size_intent, action, 
                                kb, has_reservation, col_seq, truth_seq, 
                                SQL_YN, turn_gate, args=args)
        if turn_gate != -1:
            new_str = ' '.join(map(lambda x: self.prior_corpus.dictionary.idx2word[x], new_history))
        new_str = new_str.split('<t')[-2][2:].strip().replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',')
        for i in range(1000, 1030):
            new_str = new_str.replace('<fl_'+str(i)+'>', str(i))
        terminate = False
        if new_str[-len('<eod>'):] == '<eod>':
            terminate = True
            new_str = new_str[:-len('<eod>')].strip()
        return ('Agent' if utterances[-1][0] == 'Customer' else 'Customer', new_str), terminate, predict_action
    
    def converse(self, args, model, dataloader, resume=False, save_dir='runs/exp'):
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.model_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model.load_state_dict(resume_checkpoint.model)
            self.optimizer = None
            self.args = args
            model.args = args
            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
            print('Resume from ', latest_checkpoint_path)
            print('start_epoch : ', start_epoch)
            print('step : ', step)
            start_epoch = 1
            step = 0
        else:
            print('Please Resume !')
            raise
        for batch_idx, (prior_items, selfplay_items) in enumerate(dataloader):
            print(batch_idx, prior_items[0])
            terminate = False
            utterances = []
            while not terminate:
                utterances.append(('Customer', raw_input('Customer: ')))
                next_utterance, terminate, predict_action = self.converse_one(utterances=utterances, model=model, prior_items=prior_items, selfplay_items=selfplay_items, args=args)
                print('Agent: ' + next_utterance[1])
                utterances.append(next_utterance)
            print(predict_action)
            break
        return model
    
    def reply_one(self, args, model, data_list, data_idx, utterances):
        # latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.model_dir)
        # resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        # model.load_state_dict(resume_checkpoint.model)
        # self.optimizer = None
        # self.args = args
        # model.args = args
        prior_items, selfplay_items = data_list[data_idx]
        next_utterance, terminate, predict_action = self.converse_one(utterances=utterances, model=model, prior_items=prior_items, selfplay_items=selfplay_items, args=args)
        utterances.append(next_utterance)
        return utterances, terminate, predict_action
    
    def main(self, args, model, dataloader, resume=False, save_dir='runs/exp'):
        self.converse(args, model, dataloader, resume=resume, save_dir=save_dir)
        return None