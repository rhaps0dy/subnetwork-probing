import IPython
import torch
import torch.nn.functional as F
import numpy as np

from util import from_numpy, partial_state_dict
from classifiers import POSModel, NERModel, UDModel
from subnetwork_datasets import load_conllu, build_vocab, sent_avgs, masked_loss, evaluate, load_ner
from transformer_lens.ioi_dataset import IOIDataset

N = 100

def logit_diff_from_ioi_dataset(
    logits: torch.Tensor, tokens: torch.Tensor, mean=False,
):
    assert tokens.shape == (N, 16), tokens.shape # TODO check this is not breaking things...
    assert len(logits.shape) == 3, logits.shape

    io_labels = tokens[:, 2]
    s_labels = tokens[:, 4]

    io_logits = logits[torch.arange(N), -2, io_labels]
    s_logits = logits[torch.arange(N), -2, s_labels]

    logit_diff = io_logits - s_logits
    if mean:
        return logit_diff.mean()
    else:
        return logit_diff



def train_ioi(gpt2, lambda_init = 1000, lambda_final = 10000,
              lr_base = 3e-5, mask_lr_base = 0.1, lr_warmup_frac = 0.1,
              epochs = 3, batch_size = 32, verbose = True,
              lambda_startup_frac = 0.25, lambda_warmup_frac = 0.5, subbatch_size = 8,
              masked = True):
    def calc_dev(): 
        gpt2.eval()
        all_preds = np.array([])
        all_labels = pack_labels([exmp['labels'] for exmp in dev_data])
        for j in range(0, len(dev_data), batch_size):
            exmps = dev_data[j:j+batch_size]
            sents = [exmp['sent'] for exmp in exmps]
            with torch.no_grad():
                pred = gpt2.predict_batch(sents).cpu().numpy() # numsent x 3
            pred = np.argmax(pred, axis = 1)
            all_preds = np.concatenate((all_preds, pred))
        return calc_f1(all_preds, all_labels)

    def pack_labels(labels):
        return np.array(sum(labels, []))

    def calc_f1(preds, labels):
        assert len(preds) == len(labels)
        preds = [gpt2.i2tag[pred] for pred in preds]
        labels = [gpt2.i2tag[label] for label in labels]
        result = evaluate(labels, preds)
        return result[2] / 100

    def converged(log, k = 10, thresh = 0.01, min_epochs = 25):
        if len(log) < min_epochs:
            return False
        accs = [l['dev_acc'] for l in log[-k:]]
        accs_converged = (np.max(accs) - np.min(accs)) < thresh
        if masked:
            sparsity = [l['reg_val'] for l in log[-k:]]
            sparsity_converged = (np.max(sparsity) - np.min(sparsity)) < thresh
            return accs_converged and sparsity_converged
        return accs_converged

    # blackbox this bit
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1,)
    patch_dataset = (
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
        .gen_flipped_prompts(("S", "RAND"))
        .gen_flipped_prompts(("S1", "RAND"))
    )
    train_data = ioi_dataset.toks.long() 
    # import pdb; pdb.set_trace()
    # train_data = load_ner(train_path, gpt2.tag2i)
    # dev_data = load_ner(dev_path, gpt2.tag2i)    

    print("lambda_init: {}, lambda_final: {}, lambda_startup_frac: {}, lambda_warmup_frac: {}".format(
        lambda_init, lambda_final, lambda_startup_frac, lambda_warmup_frac))
    print("lr_base: {}, mask_lr_base: {}, lr_warmup_frac: {}, epochs: {}, batch_size: {}".format(
        lr_base, mask_lr_base, lr_warmup_frac, epochs, batch_size))

    # group parameters for different learning rates

    # one parameter per thing that is masked
    mask_params = [p for n, p in gpt2.named_parameters() if 'mask_scores' in n and p.requires_grad]
    # parameters for the probe (we don't use a probe)
    gpt2_params = [p for n, p in gpt2.named_parameters() if 'mask_scores' not in n and p.requires_grad]

    assert len(gpt2_params) == 0, ("GPT2 should be empty", gpt2_params)

    trainer = torch.optim.Adam([
        {'params': mask_params, 'lr': 0., 'lr_base': mask_lr_base, 'name': 'mask'},
        {'params': gpt2_params, 'lr': 0., 'lr_base': lr_base, 'name': 'gpt2'},], lr = 0.)
    
    def set_lr(lr_ratio):
        for param_group in trainer.param_groups:
            param_group['lr'] = param_group['lr_base'] * lr_ratio

    log = []
    processed = 0
    check_processed = 0
    check_every = 2048
    lambda_reg = lambda_init
    for epoch in range(epochs):#tqdm.notebook.tqdm(range(epochs)):
        gpt2.train()
        trainer.zero_grad()
        # compute loss, also log other metrics
        loss = -1.0 * logit_diff_from_ioi_dataset(gpt2(train_data), train_data, mean=True)
        # import pdb; pdb.set_trace()
        loss.backward()
        
        if masked:
            reg = gpt2.gpt2.compute_total_regularizer()
            (lambda_reg * reg).backward()
        #mask_grad_norm = torch.nn.utils.clip_grad_norm_(mask_params, np.inf)
        #gpt2_grad_norm = torch.nn.utils.clip_grad_norm_(gpt2_params, np.inf)
        trainer.step()

        # processed += len(examples)
        # check_processed += len(examples)

        # warmup from 0 to lr_base for lr_warmup_frac
        lr_ratio = min(1, epoch / (lr_warmup_frac * epochs * len(train_data)))
        set_lr(lr_ratio)
        
        # schedule lambda_reg - constant, then linear, then constant
        lambda_ratio = max(0, min(1, (epoch - lambda_startup_frac * epochs * len(train_data)) / (lambda_warmup_frac * epochs * len(train_data))))
        lambda_reg = lambda_init + (lambda_final - lambda_init) * lambda_ratio

        if masked:
            log.append({'dev_acc': calc_dev(),
                        'loss_val': loss.item(), 
                        # 'reg_val': reg.item(), # TODO add reg

                        #'mask_grad_norm': mask_grad_norm.item(), 
                        #'gpt2_grad_norm': gpt2_grad_norm.item(), 
                        # 'pct_binary': gpt2.gpt2.compute_binary_pct()
                        })
        else:
            log.append({'dev_acc': calc_dev(),
                        'loss_val': loss.item()})
        check_processed -= check_every
        if converged(log):
            break
        if verbose:
            print("Log: {}".format(log[-1]))
        if converged(log):
            break
    return log, gpt2

def save_mask_scores(model, log, base = '../test/mask_scores'):
    keys = ['dev_acc', 'reg_val']
    fname = base
    for key in keys:
        if key in log[-1].keys():
            fname = fname + '_{}={:.4f}'.format(key, log[-1][key]*100)
    fname = fname + '.pt'

    print("Saving to {}...".format(fname))
    torch.save(partial_state_dict(model.state_dict()), fname)
    print("Done saving")
