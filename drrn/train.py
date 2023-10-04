import os
import time
import jericho
import logger
import argparse
import logging
import json
import subprocess
from env import JerichoEnv
from jericho.util import clean
from random import choice
from collections import defaultdict
import numpy as np
from lm import *
from drrn import *
from lm_finetuner import Trajectory, LM_FineTuner
logging.getLogger().setLevel(logging.CRITICAL)


def configure_logger(log_dir, add_tb=1, add_wb=1, args=None):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    log_types = [logger.make_output_format('log', log_dir), logger.make_output_format('json', log_dir),
                 logger.make_output_format('stdout', log_dir)]
    if add_tb: log_types += [logger.make_output_format('tensorboard', log_dir)]
    if add_wb: log_types += [logger.make_output_format('wandb', log_dir, args=args)]
    tb = logger.Logger(log_dir, log_types)
    global log
    log = logger.log


def train(agent, lm_finetuner, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq, args):

    # track the max state encode length
    max_state_length = 0

    start = time.time()
    obs, rewards, dones, infos, transitions = [], [], [], [], []
    env_steps, max_score, d_in, d_out = 0, 0, defaultdict(list), defaultdict(list)
    
    if lm_finetuner.lm_drop_inadmissible:  # small pre-trained classifier to filter invalid actions that CALM generates
        import fasttext
        detector = fasttext.load_model('valid_model.bin')
        reject = lambda ob: detector.predict(clean(ob))[0][0] == '__label__invalid'

    
    for env in envs:
        ob, info = env.reset()
        obs, rewards, dones, infos, transitions = obs + [ob], rewards + [0], dones + [False], infos + [info], transitions + [[]]
    states = build_state(lm_finetuner.lm, obs, infos)

    # track state encode length
    # max_state_length = max([max_state_length] + [len(state.state) for state in states])

    valid_ids = [[lm_finetuner.lm.act2ids(a) for a in info['valid']] for info in infos]

    for step in range(1, max_steps + 1):
        # act
        action_ids, action_idxs, action_values = agent.act(states, 
                                                           valid_ids, 
                                                           lm=lm_finetuner.lm, 
                                                           eps=args.eps, 
                                                           alpha=args.alpha, 
                                                           k=args.eps_top_k)
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]

        
        # log envs[0] 
        examples = [(action, value) for action, value in zip(infos[0]['valid'], action_values[0].tolist())]
        examples = sorted(examples, key=lambda x: -x[1])
        log('State  {}: {}'.format(step, lm_finetuner.lm.tokenizer.decode(states[0].state)))
        log('Actions{}: {}'.format(step, [action for action, _ in examples]))
        log('Qvalues{}: {}'.format(step, [round(value, 2) for _, value in examples]))
        # Also log true valid actions
        log('TrueActs{}:{}'.format(step, infos[0]['true_valid']))

        next_obs, next_rewards, next_dones, next_infos = [], [], [], []
        for i, (env, action) in enumerate(zip(envs, action_strs)):
            if dones[i]:

                # Build the trajectory, and push it into the finetuner
                print("----- Step {}, finish traj {}".format(step, i))
                if (infos[i]['score'] > 0) and (infos[i]['score'] >= lm_finetuner.score_threshold):
                    lm_finetuner.push(Trajectory(transitions[i], infos[i]['score'], test=False))
                else:
                    print("Traj score {}, threshold {}, do not push!".format(infos[i]['score'], lm_finetuner.score_threshold))
                print("-----")

                if env.max_score >= max_score:  # put in alpha queue
                    for transition in transitions[i]:
                        agent.observe(transition, is_prior=True)
                env_steps += infos[i]['moves']
                ob, info = env.reset()
                action_strs[i], action_ids[i], transitions[i] = 'reset', [], []
                next_obs, next_rewards, next_dones, next_infos = next_obs + [ob], next_rewards + [0], next_dones + [False], next_infos + [info]
                continue
            prev_inv, prev_look = infos[i]['inv'], infos[i]['look']
            ob, reward, done, info = env.step(action)
            
            # deal with rejection
            if lm_finetuner.lm_drop_inadmissible and step < lm_finetuner.lm_drop_threshold:
                key = hash(tuple(states[i][0] + states[i][1] + states[i][2])) # obs / desc / inv
                l_in, l_out = d_in[key], d_out[key]
                actions = infos[i]['valid']
                rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']

                # while action is invalid, pull another action from CALM generated candidates
                while not done and rej and len(actions) > 1:
                    if action not in l_out: l_out.append(action)
                    actions.remove(action)
                    action = choice(actions)
                    ob, reward, done, info = env.step(action)
                    rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']
                action_strs[i] = action

                if not rej and action not in l_in: l_in.append(action)
                if reward < 0 and action not in l_out: l_out.append(action)  # screen negative-reward actions

            next_obs, next_rewards, next_dones, next_infos = next_obs + [ob], next_rewards + [reward], next_dones + [done], next_infos + [info]
            if info['score'] > max_score:  # new high score experienced
                max_score = info['score']
                agent.memory.clear_alpha()
            if done: tb.logkv_mean('EpisodeScore', info['score'])
        rewards, dones, infos = next_rewards, next_dones, next_infos

        # continue to log envs[0]
        log('>> Action{}: {}'.format(step, action_strs[0]))
        log("Reward{}: {}, Score {}, Done {}\n".format(step, rewards[0], infos[0]['score'], dones[0]))

        # generate valid actions
        if lm_finetuner.lm_state_length_threshold > 0:
            next_states = build_state_with_threshold(lm_finetuner.lm, 
                                                     next_obs, 
                                                     infos, 
                                                     obs, 
                                                     action_strs,
                                                     lm_finetuner.lm_state_length_threshold)
        else:
            next_states = build_state(lm_finetuner.lm, next_obs, infos, prev_obs=obs, prev_acts=action_strs)
        # track state encode length
        max_state_length = max([max_state_length] + [len(state.state) for state in next_states])

        # Generate actions here
        lm_k_list, lm_k_max_list, lm_k_mean_list, lm_k_min_list, lm_k_top5_list  = [], [], [], [], []
        for env, info, state, done in zip(envs, infos, next_states, dones):
            if not done:
                # During warmup, generate 30 actions
                if step < lm_finetuner.lm_k_start_at:
                    actions, scores = lm_finetuner.lm.generate_warmup(state.state, lm_finetuner.lm_k_max)
                # During adaptive, generate 30 actions first, then conduct pruning!
                else:
                    actions, scores = lm_finetuner.lm.generate(state.state, lm_finetuner.lm_k_max)

                # if not len(actions) == len(scores):
                #     actions = actions[:len(scores)]

                # reframe scores
                scores = [-1. / score for score in scores]
                prev_sum = sum(scores)
                scores = [score / prev_sum for score in scores]

                if step >= lm_finetuner.lm_k_start_at:
                    while (sum(scores) > lm_finetuner.lm_k_thres) and (len(scores) > lm_finetuner.lm_k_min):
                        actions.pop()
                        scores.pop()

                lm_k_list.append(len(actions))
                lm_k_max_list.append(np.max(scores))
                lm_k_mean_list.append(np.mean(scores))
                lm_k_min_list.append(np.min(scores))
                lm_k_top5_list.append(np.array(scores)[np.argpartition(scores, -5)[-5:]].sum())
                # print("#Acts {}|Max {:.3f}|Mean {:.3f}|Min {:.3f}|Top5Sum {:.3f}".format(lm_k_list[-1], lm_k_max_list[-1], lm_k_mean_list[-1], lm_k_min_list[-1], lm_k_top5_list[-1]))

                if lm_finetuner.lm_drop_inadmissible and step < lm_finetuner.lm_drop_threshold:
                    key = hash(tuple(state[0] + state[1] + state[2]))
                    l_in, l_out = d_in[key], d_out[key]
                    actions += [action for action in l_in if action not in actions]  # add extra valid
                    actions = [action for action in actions if action and action not in l_out]  # remove invalid
                if not actions: actions = ['wait', 'yes', 'no']
                info['valid'] = list(set(actions))

        
        next_valids = [[lm_finetuner.lm.act2ids(a) for a in info['valid']] for info in infos]
        for state, act, rew, next_state, valids, done, transition in zip(states, 
                                                                         action_ids, 
                                                                         rewards, 
                                                                         next_states,
                                                                         next_valids, 
                                                                         dones, 
                                                                         transitions):
            if act:  # not [] (i.e. reset)
                transition.append(Transition(state, act, rew, next_state, valids, done))
                agent.observe(transition[-1])  # , is_prior=(rew != 0))
        obs, states, valid_ids = next_obs, next_states, next_valids

        if step % log_freq == 0:
            tb.logkv('Step', step)
            tb.logkv('Env Steps', env_steps)
            tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
            tb.logkv("MaxScore", max_score)
            tb.logkv("#dict", len(lm_finetuner.lm.generate_dict))
            tb.logkv("MaxStateLength", max_state_length)
            tb.logkv("LM_K", np.mean(lm_k_list))
            tb.logkv("LM_K_MaxScore", np.mean(lm_k_max_list))
            tb.logkv("LM_K_MeanScore", np.mean(lm_k_mean_list))
            tb.logkv("LM_K_MinScore", np.mean(lm_k_min_list))
            tb.logkv("LM_K_Top5ScoreSum", np.mean(lm_k_top5_list))
            tb.logkv("EpisodeScores100", sum(env.get_end_scores(last=100) for env in envs) / len(envs))
            
            print("LM_K:              {}".format(["{:5d}".format(item) for item in lm_k_list]))
            print("LM_K_MaxScore:     {}".format(["{:.3f}".format(item) for item in lm_k_max_list]))
            print("LM_K_MeanScore:    {}".format(["{:.3f}".format(item) for item in lm_k_mean_list]))
            print("LM_K_MinScore:     {}".format(["{:.3f}".format(item) for item in lm_k_min_list]))
            print("LM_K_Top5ScoreSum: {}".format(["{:.3f}".format(item) for item in lm_k_top5_list]))

            if len(lm_finetuner.lm_ft_buffer) > 0:
                tb.logkv("LMFT BufferMaxScore", np.max([item.score for item in lm_finetuner.lm_ft_buffer]))
                tb.logkv("LMFT BufferMeanScore", np.mean([item.score for item in lm_finetuner.lm_ft_buffer]))
                tb.logkv("LMFT BufferMeanLength", -np.mean([item.neg_length for item in lm_finetuner.lm_ft_buffer]))
            tb.dumpkvs()
        if step % update_freq == 0:
            loss = agent.update()
            if loss is not None:
                tb.logkv_mean('Loss', loss)
        if step % checkpoint_freq == 0:
            json.dump(d_in, open('%s/d_in.json' % args.output_dir, 'w'), indent=4)
            json.dump(d_out, open('%s/d_out.json' % args.output_dir, 'w'), indent=4)
            json.dump(lm_finetuner.lm.generate_dict, open('%s/lm.json' % args.output_dir, 'w'), indent=4)

        # [20220605] fine-tune and validate
        if (step % lm_finetuner.lm_ft_freq == 0) and (step >= lm_finetuner.lm_ft_start_at) and (step < lm_finetuner.lm_ft_stop_at) and (len(lm_finetuner.lm_ft_buffer) >= lm_finetuner.lm_ft_buffer_min_start):
            print("===== Step {}, conduct FT and Val".format(step))

            lm_ft_acc, lm_ft_loss = lm_finetuner.finetune()
            tb.logkv('LMFT Acc', lm_ft_acc)
            tb.logkv('LMFT Loss', lm_ft_loss)
            lm_val_acc, lm_val_loss = lm_finetuner.validate()
            tb.logkv('LMVal Acc', lm_val_acc)
            tb.logkv('LMVal Loss', lm_val_loss)

            lm_val_preca, lm_val_reca, lm_val_recg = lm_finetuner.validate_interaction()
            tb.logkv('LMVal PrecA', lm_val_preca)
            tb.logkv('LMVal RecA', lm_val_reca)
            tb.logkv('LMVal RecG', lm_val_recg)

            val_progress = "FT Acc {:.3f}|FT Loss {:.3f}|Val Acc {:.3f}|Val Loss {:.3f}|Val PrecA {:.3f}|Val RecA {:.3f}|Val RecG {:.3f}"
            val_progress = val_progress.format(lm_ft_acc, lm_ft_loss,
                                               lm_val_acc, lm_val_loss,
                                               lm_val_preca, lm_val_reca, lm_val_recg)
            print(val_progress)

            # print("===== ===== =====")
            # print("Exit here!")
            # exit()



def parse_args():
    parser = argparse.ArgumentParser()
    #  Here "st20k" means that I start both LMFT and AdaptiveK from the 20k steps
    parser.add_argument('--output_dir', default='./logs/csm') 
    parser.add_argument('--rom_path', default='../games/snacktime.z8')             
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)                 
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=1000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)
    parser.add_argument('--priority_fraction', default=0.5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)

    # logger
    parser.add_argument('--tensorboard', default=0, type=int)
    parser.add_argument('--wandb', default=0, type=int)

    # LM general
    parser.add_argument('--lm_path', default='../download-models/gpt2') 

    # LM adaptive K
    parser.add_argument('--lm_k_max', default=30, type=int)   
    parser.add_argument('--lm_k_min', default=10, type=int)   
    parser.add_argument('--lm_k_thres', default=0.6, type=float, help='the cumulative score of k actions')  
    parser.add_argument('--lm_k_start_at', default=20000, type=int, help='when to start adaptiveK') # 0               
    
    # LM drop inadmissible
    parser.add_argument('--lm_drop_inadmissible', default=1, type=int) # Whether or not to drop the inadmissible actions (fasttext
    parser.add_argument('--lm_drop_threshold', default=20000, type=int) # When to stop dropping inadmissible actions
    parser.add_argument('--lm_state_length_threshold', default=100, type=int) # Whether or not to use the length threshold

    # LM FT
    parser.add_argument('--lm_ft_freq', default=500, type=int, help='fine-tuning frequency') # 2000
    parser.add_argument('--lm_ft_val_path', default="../download-models/jericho_walkthrough_data/walkthrough_snacktime.json")
    parser.add_argument('--lm_ft_start_at', default=20000, type=int, help='when to start fine-tuning, e.g. after 20000 steps') # 0
    parser.add_argument('--lm_ft_stop_at', default=10000000, type=int, help='when to stop fine-tuning, e.g. after 20000 steps') # 10000000
    parser.add_argument('--lm_ft_buffer_size', default=50, type=int, help='the FT buffer size') # 50
    parser.add_argument('--lm_ft_buffer_min_start', default=20, type=int, help='the min buffer size for starting FT') # 20
    parser.add_argument('--lm_ft_thres_type', default='pos', type=str, help='how to determine the threshold for pushing, max / mean / pos') # max
    parser.add_argument('--lm_ft_epoch', default=3, type=int, help='the number of epochs per FT') # 3
    parser.add_argument('--lm_ft_batch_size', default=8, type=int, help='the FT batch size')      # 8


    # exploration -> useless in our work
    parser.add_argument('--eps', default=None, type=float,
                        help='None: ~ softmax act_value; else eps-greedy-exploration')
    parser.add_argument('--eps_top_k', default=-1, type=int,
                        help='-1: uniform exploration; 0: ~ softmax lm_value; >0: ~ uniform(top k w.r.t. lm_value)')
    parser.add_argument('--alpha', default=0, type=float,
                        help='act_value = alpha * bert_value + (1-alpha) * q_value; only used when eps is None now')

    return parser.parse_args()


def main():
    assert jericho.__version__.startswith('3'), "This code is designed to be run with Jericho version >= 3.0.0."
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Initialize LM")
    language_model = GPT2LM(args.lm_path)
    args.vocab_size = len(language_model.tokenizer)

    print("=== Initialize LM finetuner")
    lm_finetuner = LM_FineTuner(lm=language_model, args=args)

    print("=== Initialize logger")
    print(args)
    configure_logger(args.output_dir, args.tensorboard, args.wandb, args)

    print("=== Initialize RL agent")
    agent = DRRN_Agent(args)

    print("=== Initialize env")
    envs = [JerichoEnv(rom_path=args.rom_path, 
                       seed=args.seed, 
                       step_limit=args.env_step_limit, 
                       get_valid = False) 
            for _ in range(args.num_envs)] # Do not allow getting valid actions from the env
    
    print("Start training")

    lm_val_acc, lm_val_loss = lm_finetuner.validate()
    lm_val_preca, lm_val_reca, lm_val_recg = lm_finetuner.validate_interaction()
    val_progress = "Initial LM Val Acc {:.3f}|Val Loss {:.3f}|Val PrecA {:.3f}|Val RecA {:.3f}|Val RecG {:.3f}"
    val_progress = val_progress.format(lm_val_acc, lm_val_loss, lm_val_preca, lm_val_reca, lm_val_recg)
    print(val_progress)
    # exit()

    train(agent, lm_finetuner, envs, args.max_steps, args.update_freq, args.eval_freq, args.checkpoint_freq, args.log_freq, args)
    for env in envs:
        env.close()

if __name__ == "__main__":
    main()
