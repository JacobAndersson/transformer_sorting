import torch
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig
from dataclasses import dataclass

import pickle as pkl
import fire

@dataclass
class TrainConfig:
    list_length: int = 5
    num_samples: int = 10000
    batch_size: int = 100
    min_value: int = 0
    max_value: int = 64
    bos_token: int = 65
    mid_token: int = 66
    test_seed: int = 62
    train_seed: int = 42
    var_length: bool = False

def calc_accuracy(batch):
    list_length = int(batch.shape[1] /2 - 1)
    unsorted = batch[:, 1:list_length+1].tolist()
    predicted = batch[:, list_length+2:].tolist()
    
    tot_correct = 0
    tot_el = 0
    for (un, pred) in zip(unsorted, predicted):
        tot_correct += sum([x == y for x,y in zip(pred, sorted(un))])
        tot_el += len(un)
        
    return tot_correct/tot_el, (tot_correct, tot_el)

def generate_batches(cfg, dataset):
    if dataset == 'train':
        seed = cfg.train_seed
        samples = cfg.num_samples
    elif dataset == 'test':
        seed = cfg.test_seed
        samples = cfg.num_samples
    else:
        seed = 42
        samples = 10000

    torch.manual_seed(seed)
    random_list = torch.randint(cfg.min_value, cfg.max_value, (samples, cfg.list_length)).tolist()
    lists = []
    for entry in random_list:
        entry = [cfg.bos_token] + entry + [cfg.mid_token] + sorted(entry)
        lists.append(entry)

    batches = torch.tensor(lists)

    batches = torch.split(batches, cfg.batch_size)
    return batches

def generate_var_length_batches(cfg, dataset):
    if dataset == 'train':
        seed = cfg.train_seed
        samples = cfg.num_samples
    elif dataset == 'test':
        seed = cfg.test_seed
        samples = cfg.num_samples
    else:
        seed = 42
        samples = 10000

    torch.manual_seed(seed)

    #random_list = torch.randint(cfg.min_value, cfg.max_value, (samples, cfg.list_length)).tolist()
    num_batches = samples // cfg.batch_size

    batches = []
    for _ in range(num_batches):
        length = torch.randint(1, cfg.list_length+1, (1,)).item()
        batch = []
        for _ in range(cfg.batch_size):
            random_list = torch.randint(cfg.min_value, cfg.max_value, (1, length)).tolist()[0]
            entry = [cfg.bos_token] + random_list + [cfg.mid_token] + sorted(random_list)
            batch.append(entry)
        batches.append(torch.tensor(batch))

    return batches

def loss_fn(out, target, list_length):
    # Strip away everything before the mid token. Loss should not be calculated
    # for the unsorted part of the list.
    logits = out[:, list_length+1:-1]
    target = target[:, list_length+2:]

    # Calc the probabilities for each token 
    probs = torch.log_softmax(logits, dim=-1)

    # Get the probability for the correct token
    correct_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return -correct_probs.mean()

def predict(out, list_length):
    logits = out[:, list_length+1:-1]
    probs = torch.softmax(logits, dim=-1)
    return probs.argmax(-1)

def main(
    list_length: int = 5,
    num_samples: int = 10000,
    batch_size: int = 100,
    min_value: int = 0,
    max_value: int = 64,
    var_length: bool = False,
    pth: str = 'model.pkl',
    num_epochs: int = 3,
    ):
    cfg = HookedTransformerConfig(
        d_model=128,
        n_layers=1,
        n_heads=1,
        d_head=128,
        n_ctx=2*list_length + 2,
        d_vocab=max_value + 3,
        act_fn='relu',
        attn_only=True,
        device='cpu',
        seed=42,
        normalization_type=None
    )

    bos_token = max_value + 1
    mid_token = max_value + 2

    train_cfg = TrainConfig(
        list_length=list_length,
        num_samples=num_samples,
        batch_size=batch_size,
        min_value=min_value,
        max_value=max_value,
        bos_token=bos_token,
        mid_token=mid_token,
        test_seed=62,
        train_seed=42,
        var_length=var_length,
    )

    if not var_length:
        data = generate_batches(train_cfg, 'train')
        test_data = generate_batches(train_cfg, 'test')
    else:
        data = generate_var_length_batches(train_cfg, 'train')
        test_data = generate_var_length_batches(train_cfg, 'test')

    model = HookedTransformer(cfg)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(.9, .999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode='min',
    )

    def viz(d):
        list_length = d[0].shape[1] // 2 - 1

        ex = d[0]
        unsorted = ex[:, 1:list_length+1]
        out = model(ex)
        pred = predict(out, list_length)

        for (i, (u, p)) in enumerate(zip(unsorted, pred)):
            u = u.tolist()
            p = p.tolist()
            print(f'{i}: {u} -> {p}')

    def generation_acc(d):
        list_length = d[0].shape[1] // 2 - 1
        
        tot_correct = 0
        tot_el = 0
        for batch in d:
            clipped = batch[:, :list_length+2]

            predicted_sort = model.generate(clipped,
                                      max_new_tokens=list_length,
                                      stop_at_eos=False,
                                      prepend_bos=False,
                                      verbose=False,
                                      )
            acc, (correct, total)= calc_accuracy(predicted_sort)
            tot_correct += correct
            tot_el += total
        
        acc = tot_correct / tot_el
        print(f'Accuracy: {tot_correct / tot_el}, {tot_correct}/{tot_el}')
        return acc

    for epoch in range(num_epochs):
        for i, batch in enumerate(data):
            out = model(batch)

            if (var_length):
                list_length = batch.shape[1] // 2 - 1 

            loss = loss_fn(out, batch, list_length)
            loss.backward()
            optim.step()
            optim.zero_grad()

            if i % 50 == 0:
                total_test_loss = 0.0
                for test_batch in test_data:
                    out = model(test_batch)
                    loss = loss_fn(out, test_batch, list_length)
                    total_test_loss += loss.item()

                gen_acc = generation_acc(test_data)
                print('####')
                print('epoch', epoch)
                print('test loss', total_test_loss / len(test_data))
                print('generation acc', gen_acc)
                print('####')
                scheduler.step(total_test_loss / len(test_data))

    #viz(test_data)
    print('Saving model to', pth)
    with open(pth, 'wb') as f:
        pkl.dump({
            "model": model.state_dict(),
            "cfg": cfg,
            "train_cfg": train_cfg,
            }, f)

    print(model)

if __name__ == '__main__':
    fire.Fire(main)
