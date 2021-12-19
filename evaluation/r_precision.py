import torch
import sys, os, pickle
import numpy as np
from torchvision import transforms

sys.path.append('..')
from evaluation.model import CNN_ENCODER, RNN_ENCODER
from utils.data_utils import CUBDataset


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

from miscc.config import cfg, cfg_from_file
cfg_from_file('../cfg/eval_birds.yml') #train

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Returns cosine similarity between x1 and x2, computed along dim
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))

def prepare_data(imgs, captions, captions_lens, class_ids, keys, captions_idx):
    # sort data by the length in a decreasing order
    # the reason of sorting data can be found in https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        real_imgs.append(imgs[i])

    sorted_captions = captions[sorted_cap_indices].squeeze()
    if captions.size(0) == 1:
        captions = captions.unsqueeze(0)
    sorted_class_ids = class_ids[sorted_cap_indices].numpy()
    sorted_keys = [keys[i] for i in sorted_cap_indices.numpy()]
    sorted_captions_idx = captions_idx[sorted_cap_indices].numpy()

    return [real_imgs, sorted_captions, sorted_cap_lens, sorted_class_ids, sorted_keys, sorted_captions_idx]

image_encoder = CNN_ENCODER(256)
state_dict = torch.load('./sim_models/bird/image_encoder.pth', map_location=lambda storage, loc: storage)
image_encoder.load_state_dict(state_dict)
for p in image_encoder.parameters():
    p.requires_grad = False
print('Load image encoder')
image_encoder.eval()

# load the image encoder model to obtain the latent feature of the real caption
text_encoder = RNN_ENCODER(5450, nhidden=256)
state_dict = torch.load('./sim_models/bird/text_encoder.pth', map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
print('Load text encoder')
text_encoder.eval()

image_encoder = image_encoder.to(device)
text_encoder = text_encoder.to(device)

transform = transforms.Compose([
    transforms.Resize((128, 128))
])

test_dataset = CUBDataset(cfg.DATA_DIR, transform=transform, split='test', eval_mode=True)

print(f'\ttest data directory:\n{test_dataset.split_dir}\n')
print(f'\t# of test filenames:{test_dataset.filenames.shape}\n')
print(f'\texample of filename of test image:{test_dataset.filenames[0]}\n')
print(f'\texample of caption and its ids:\n{test_dataset.captions[0]}\n{test_dataset.captions_ids[0]}\n')
print(f'\t# of test captions:{np.asarray(test_dataset.captions).shape}\n')
print(f'\t# of test caption ids:{np.asarray(test_dataset.captions_ids).shape}\n')

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                                              drop_last=False, shuffle=False, num_workers=int(cfg.WORKERS))

test_dataset_for_wrong = CUBDataset(cfg.DATA_DIR, transform=transform, split='test', eval_mode=True, for_wrong=True)
current_dir = os.getcwd()
fname = os.path.join(current_dir.replace('/evaluation', ''), cfg.DATA_DIR, 'test', 'test_caption_info.pickle')
with open(fname, 'rb') as f:
    test_caption_info = pickle.load(f)

n_data = np.asarray(test_dataset.captions_ids).shape[0]
true_cnn_features = np.zeros((n_data, cfg.TEXT.EMBEDDING_DIM), dtype=float)
true_rnn_features = np.zeros((n_data, cfg.TEXT.EMBEDDING_DIM), dtype=float)
wrong_rnn_features = np.zeros((n_data, cfg.WRONG_CAPTION, cfg.TEXT.EMBEDDING_DIM), dtype=float)

total_cs = []
cnn_features = []
for batch_idx, data in enumerate(test_dataloader):
    gen_imgs = data['gen_img']
    captions = data['caps']
    captions_lens = data['cap_len']
    class_ids = data['cls_id']
    keys = data['key']
    captions_idx = data['cap_ix']

    sorted_gen_imgs, sorted_captions, sorted_cap_lens, sorted_class_ids, sorted_keys, sorted_captions_idx = prepare_data(
        gen_imgs, captions, captions_lens, class_ids, keys, captions_idx)

    if cfg.CUDA:
        sorted_captions = sorted_captions.to(device)
        sorted_cap_lens = sorted_cap_lens.to(device)

    hidden = text_encoder.init_hidden(sorted_captions.size(0))
    _, sent_emb = text_encoder(sorted_captions, sorted_cap_lens, hidden)

    _, gen_sent_code = image_encoder(sorted_gen_imgs[-1].to(device))

    true_sim = cosine_similarity(gen_sent_code, sent_emb)

    true_cnn_features[captions_idx] = gen_sent_code.detach().cpu().numpy()
    true_rnn_features[captions_idx] = sent_emb.detach().cpu().numpy()

    false_sim_list = []
    for i in range(captions.size(0)):
        assert sorted_class_ids[i] == test_caption_info[sorted_captions_idx[i]][0]
        cap_cls_id = test_caption_info[sorted_captions_idx[i]][1]
        mis_match_captions, sorted_mis_cap_lens = test_dataset_for_wrong.get_mis_captions(sorted_class_ids[i],
                                                                                          cap_cls_id)

        if cfg.CUDA:
            mis_match_captions = mis_match_captions.to(device)
            sorted_mis_cap_lens = sorted_mis_cap_lens.to(device)

        mis_hidden = text_encoder.init_hidden(mis_match_captions.size(0))
        _, mis_sent_emb = text_encoder(mis_match_captions, sorted_mis_cap_lens, mis_hidden)

        false_sim = cosine_similarity(gen_sent_code[i:i + 1], mis_sent_emb)

        wrong_rnn_features[captions_idx] = mis_sent_emb.detach().cpu().numpy()

        false_sim_list.append(false_sim)

    batch_cs = torch.cat([torch.unsqueeze(true_sim, 1), torch.stack(false_sim_list, dim=0)], dim=1)
    total_cs.append(batch_cs)

total_cs = torch.cat(total_cs, dim=0)
r_precision = torch.mean((torch.argmax(total_cs, dim=1) == 0) * 1.0).cpu().detach().numpy()
print('# images for evaluation:', len(total_cs))
print('R-precision: ' + str(r_precision))

np.savez(cfg.R_PRECISION_FILE, total_cs=total_cs.detach().cpu().numpy(),
         true_cnn_features=true_cnn_features, true_rnn_features=true_rnn_features,
         wrong_rnn_features=wrong_rnn_features)
