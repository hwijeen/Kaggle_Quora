import torch

class Args():

    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    # training
    batch_size = 128
    epoch = 30

    # optimization
    pos_weight = 1.0
    threshold = 0.35
    scheduler_patience = 2
    early_stop_patience = 5

    # data
    #train_path = 'data/toy_train.csv'
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'


    # data
    max_vocab = 50000 # DELETE .PY WHEN CHANGEING THIS
    max_len = 80
    min_freq = 3

    # embedding
    embedding_type = 'pretrained'#'DME' #'unweighted'
    cache = '/home/nlpgpu5/data/embeddings/' # where to save!
    embedding = ['/home/nlpgpu5/data/embeddings/glove.840B.300d.txt']
    #embedding = ['/home/nlpgpu5/data/embeddings/glove.840B.300d.txt',
    #             '/home/nlpgpu5/data/embeddings/paragram_300_sl999.txt',
    #             '/home/nlpgpu5/data/embeddings/wiki-news-300d-1M.vec']
    freeze_emb = True
    dropout_emb = 0.33
    dim_word = 300

    # model
    model = 'rnn' # 'transformer'
    # TODO: rnn이 multihead쓸 때도 h 개수 지정 필요
    attention = 'multihead'#'location' #'dot'
    if model == 'transformer':
        lengths = False
        # TODO: dim_model 300 아니면 projection  필요
        dim_model = 300; h = 6; N = 4; dim_ff = 2048;
        dropout_transformer = 0.33; dropout_attn = 0.5

    elif model == 'rnn':
        lengths = True
        mtype = 'pool'; dim_hidden = 128;
        bidirectional = True; num_layers = 4
        dropout_rnn = 0.33; dropout_attn = 0.33

    elif model == 'cnn':
        dim_word = 300; windows = (2, 3, 4); dim_feature = 200

    # classifier
    dim_fc = 128
    dropout_fc = 0.5
