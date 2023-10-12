import torch.nn as nn
import torch.distributions as dist
from mem_transformer import MemTransformerLM

class Encoder(nn.Module):
    def __init__(self, embedding, ntoken, z_dim,  embed_dim, nhead, nhid, nlayers, gain, bptt, 
                 dropout=0., scale_emb_grad_by_freq=False):
        super(Encoder, self).__init__()
        
        self.ntoken = ntoken
        self.bptt = bptt
        self.gain = gain
        self.z_dim = z_dim
        self.nhead = nhead
        dropatt = dropout


        self.project_trans_out = nn.Linear(embed_dim, z_dim)

        #long-context encoder uses cross-segment memory for attending long sequences
        self.transformer_encoder1 = MemTransformerLM(ntoken, nlayers, nhead, embed_dim, embed_dim//nhead, nhid,
                 dropout, dropatt, tie_weight=False, d_embed=embed_dim, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=bptt, ext_len=0, mem_len=bptt, 
                 cutoffs=[], adapt_inp=False,
                 same_length=True, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1)
        
        self.temperature = nn.Parameter(torch.tensor([1/.07]).log())

        self.embedding = embedding

        self.embed_dim = embed_dim
        self.nhead = nhead

        self.init_weights()
        self.mems = None

    def erase_mems(self, mask):
      new_mems = []
      for m in self.mems:
        new_mem = m.clone()
        new_mem[..., mask, :] = 0.
        new_mems.append(new_mem)
      self.mems = new_mems


    def init_weights(self):
        projections = [self.project_c_to_joint_space, self.project_p_to_joint_space, self.project_trans_out]
        gain = self.gain
        for l in self.transformer_encoder1.layers:
          pos_ff = l.pos_ff
          ff = pos_ff.CoreNet
          for n in ff:
            if hasattr(n, "weight"):
              torch.nn.init.kaiming_normal_(n.weight.data, gain)
        for proj in projections:
          torch.nn.init.kaiming_normal_(proj.weight.data, gain)
        torch.nn.init.zeros_(self.embedding.weight)


    def _src_to_z(self, src, mems, store_mems=True, dropout=0.):
     
        trans_out, new_mems = self.transformer_encoder1._forward(src, mems, False, dropout=dropout)
        trans_out = self.project_trans_out(trans_out)
        if store_mems:
          self.mems = new_mems

        return trans_out
      
      
    def forward(self, src, store_mems=True, use_mems=True, dropout=0.):
        mems = self.mems if use_mems else None
        if not mems:
          mems = self.transformer_encoder1.init_mems()

        z = self._src_to_z(src.clone(), mems=mems, store_mems=store_mems, dropout=dropout) 

        return z
        

class UJ2Vec(nn.Module):
    def __init__(self, ntoken, bptt, z_prior_scale, dropout=0., gain = 0.1, z_dim=64, embed_dim=64, 
                 hidden_dim=512, nhead=4, nlayers=4, noutputs=1, 
                 use_cuda=False):
        super(UJ2Vec, self).__init__()
        self.z_prior_scale = z_prior_scale
        self.gain = gain
        self.nlayers = nlayers
        self.nparticles = 1
        self.ntoken = ntoken
        self.bptt = bptt
        self.z_prior_scale = z_prior_scale              
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.noutputs=noutputs

        self.prev_z = None
        self.embedding = nn.Embedding(ntoken, embed_dim, padding_idx=1, sparse=False, scale_grad_by_freq=False)
        # create the encoder and decoder networks
        self.encoder = Encoder(self.embedding, dropout=dropout, z_dim=z_dim, embed_dim=embed_dim, nhid=hidden_dim, 

                               nlayers=nlayers, nhead=nhead, ntoken=ntoken,
                               gain=gain, bptt=self.bptt)

        #Two projections from current view and a random previous view into joint space
        self.project_c_to_joint_space = nn.Linear(z_dim, z_dim)
        self.project_p_to_joint_space = nn.Linear(z_dim, z_dim)

        self.decoder_representation = [
            self.project_c_to_joint_space, 
            self.project_p_to_joint_space
        ]       
        self.decoder_prediction = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ntoken),
            nn.Sigmoid()
        )
        self.encoder.transformer_encoder1.word_emb = self.embedding
        
        if use_cuda:
            self.cuda()
 
    def contrastive_loss(self, u, v):
      """
      u: joint space embedding of current segment
      v: joint space embedding of a previous segment
    
      Computes the loss based on the cosine similarity of 
      u and v
      """
      logits = torch.einsum("...hi,...ji->...hj",u, v)
      logits = logits * torch.exp(self.encoder.temperature).clamp(1e-5, 500.)
      labels = torch.arange(u.size(-2))
    
      loss_c = dist.Categorical(logits=logits).log_prob(labels)
      loss_p = dist.Categorical(logits=logits.transpose(-1, -2)).log_prob(labels)
      loss = -(loss_c + loss_p)/2
    
      return loss
        
    def elbo_loss(self, z_loc, z_scale, y):
      """
      z_loc: Mean parameter of the variational distribution
      z_scale: standard deviation parameter of the variational distribution
    
      computes the loss given the approximate posterior distribution, q_z, and y
      """
      p_z = dist.Normal(torch.zeros_like(z_loc), torch.ones_like(z_scale))
      q_z dist.Normal(z_loc, z_scale)
        
      p = self.decoder(z)
    
      cross_entropy = dist.Categorical(probs=p).log_prob(y)
      kl_divergence = dist.kl.kl_divergence(q_z, p_z)
      
      loss = - (cross_entropy - kl_divergence)
    
      return loss

