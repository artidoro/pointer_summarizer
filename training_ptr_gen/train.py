from __future__ import unicode_literals, print_function, division

import os
import time

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm

from custom_adagrad import AdagradCustom

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
import argparse
from tqdm import tqdm

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self, args, model_name = None):
        self.args = args
        vocab = args.vocab_path if args.vocab_path is not None else config.vocab_path
        self.vocab = Vocab(vocab, config.vocab_size, config.embedding_file)
        self.batcher = Batcher(args.train_data_path, self.vocab, mode='train',
                               batch_size=args.batch_size, single_pass=False, args=args)
        self.eval_batcher = Batcher(args.eval_data_path, self.vocab, mode='eval',
                                    batch_size=args.batch_size, single_pass=True, args=args)
        time.sleep(15)

        if model_name is None:
            self.train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        else:
            self.train_dir = os.path.join(config.log_root, model_name)

        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)

        self.model_dir = os.path.join(self.train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        #self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        print(model_save_path)
        # logger.debug(model_save_path)
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(self.vocab, model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = self.args.lr_coverage if config.is_coverage else self.args.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        self.optimizer.zero_grad()

        loss = self.get_loss(batch)

        loss.backward()

        clip_grad_norm(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def get_loss(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)
        encoder_outputs, encoder_hidden, max_encoder_output = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        if config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output
        step_losses = []
        for di in range(min(max_dec_len, self.args.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                      encoder_outputs, enc_padding_mask,
                                                                                      c_t_1,
                                                                                      extra_zeros,
                                                                                      enc_batch_extend_vocab,
                                                                                      coverage)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        return loss

    def trainIters(self, n_iters, model_file_path=None):
        start_iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        best_val_loss = None
        for it in tqdm(range(start_iter, n_iters)):
            iter = start_iter + it
            self.model.train()
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
            #iter += 1

            print_interval = 1000
            if iter!=0 and iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter!= 0 and iter % 5000 == 0:
                loss = self.run_eval()
                if best_val_loss is None or loss < best_val_loss:
                    best_val_loss = loss
                    self.save_model(running_avg_loss, iter)
                    print("Saving best model")
                    # logger.debug("Saving best model")

    def run_eval(self):
        running_avg_loss, iter = 0, 0
        self.model.eval()
        self.eval_batcher._finished_reading = False
        self.eval_batcher.setup_queues()
        batch = self.eval_batcher.next_batch()
        while batch is not None:
            loss = self.get_loss(batch).item()
            if loss is not None:
                running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
                iter += 1
            batch = self.eval_batcher.next_batch()
        msg = 'Eval: loss: %f' % running_avg_loss
        print(msg)
        return running_avg_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Structured Summarization Model')
    parser.add_argument('--save_path', type=str, default=None, help='location of the save path')
    parser.add_argument('--reload_path', type=str, default=None, help='location of the older saved path')
    parser.add_argument('--train_data_path', type=str, default='/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/chunked/train_*', help='location of the train data path')
    parser.add_argument('--eval_data_path', type=str, default='/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/val.bin', help='location of the eval data path')
    parser.add_argument('--vocab_path', type=str, default=None, help='location of the eval data path')

    parser.add_argument('--lr', type=float, default=0.15, help='Learning Rate')
    parser.add_argument('--lr_coverage', type=float, default=0.15, help='Learning Rate for Coverage')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--max_dec_steps', type=int, default=100, help='Max Dec Steps')

    # parser.add_argument('--use_glove', action='store_true', default=False, help='use_glove_embeddings for training')

    # if all false - summarization with just plain attention over sentences - 17.6 or so rouge

    args = parser.parse_args()
    save_path = args.save_path
    reload_path = args.reload_path

    train_processor = Train(args, save_path)
    train_processor.trainIters(config.max_iterations, reload_path)
