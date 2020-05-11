import torch
from sklearn.metrics import f1_score
import dgl
from utils import load_data, EarlyStopping

import dataprocess_han
from args import read_args
from model import HAN
from sklearn.model_selection import train_test_split

from warnings import filterwarnings
filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from warnings import simplefilter
simplefilter(action='ignore', category=Warning)

def score(logits, labels):
  _, indices = torch.max(logits, dim=1)
  prediction = indices.long().cpu().numpy()
  labels = labels.cpu().numpy()

  accuracy = (prediction == labels).sum() / len(prediction)
  micro_f1 = f1_score(labels, prediction, average='micro')
  macro_f1 = f1_score(labels, prediction, average='macro')

  return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
  model.eval()
  with torch.no_grad():
    logits,_ = model(g, features)
  loss = loss_func(logits[mask], labels[mask])
  accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

  return loss, accuracy, micro_f1, macro_f1

def get_binary_mask(total_size, indices):
  mask = torch.zeros(total_size)
  mask[indices] = 1
  return mask.byte()

def main(args):
  # If args['hetero'] is True, g would be a heterogeneous graph.
  # Otherwise, it will be a list of homogeneous graphs.
  args_academic = read_args()
  data = dataprocess_han.input_data_han(args_academic)  
  #g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
      #val_mask, test_mask = load_data(args['dataset'])
  features = torch.tensor(data.a_text_embed,dtype=torch.float32)
  labels=torch.tensor(data.a_class)
  

  
  
  APA_g = dgl.graph(data.APA_matrix, ntype='author', etype='coauthor')
  APVPA_g = dgl.graph(data.APVPA_matrix, ntype='author', etype='attendance')
  APPA_g = dgl.graph(data.APPA_matrix, ntype='author', etype='reference')
  
  #g = [APA_g, APPA_g]
  g = [APA_g, APVPA_g, APPA_g]
  
  num_classes = 4 
  features = features.to(args['device'])
  labels = labels.to(args['device'])

  #if args['hetero']:
    #from model_hetero import HAN
    #model = HAN(meta_paths=[['pa', 'ap'], ['pf', 'fp']],
                    #in_size=features.shape[1],
                    #hidden_size=args['hidden_units'],
                    #out_size=num_classes,
                    #num_heads=args['num_heads'],
                    #dropout=args['dropout']).to(args['device'])
  #else:
  model = HAN(num_meta_paths=len(g),
                  in_size=features.shape[1],
                  hidden_size=args['hidden_units'],
                  out_size=num_classes,
                  num_heads=args['num_heads'],
                  dropout=args['dropout']).to(args['device'])

  stopper = EarlyStopping(patience=args['patience'])
  loss_fcn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
  model.load_state_dict(torch.load("./model_para.pt"))

  for epoch in range(args['num_epochs']):
    
    X=[[i] for i in range(args_academic.A_n)]
    train_X,test_X,_,_ = train_test_split(X,X,test_size=0.8) #
    train_X,test_X,_,_ = train_test_split(train_X,train_X,test_size=0.2) #
    
    train_mask = get_binary_mask(args_academic.A_n, train_X)
    test_mask = get_binary_mask(args_academic.A_n, test_X)
    
    #train_mask = torch.tensor(data.train_mask)
    #test_mask = torch.tensor(data.test_mask)
    val_mask = test_mask    
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])    
    model.train()
    logits, _ = model(g, features)
    loss = loss_fcn(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
    val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
    early_stop = stopper.step(val_loss.data.item(), val_acc, model)

    print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
                epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

    if early_stop:
      break

  stopper.load_checkpoint(model)
  model.eval()
  _, embedding = model(g, features)
  embed_file = open( "./node_embedding.txt", "w")
  for k in range(embedding.shape[0]):
    embed_file.write('a' + str(k) + " ")
    for l in range(embedding.shape[1] - 1):
      embed_file.write(str(embedding[k][l].item()) + " ")
    embed_file.write(str(embedding[k][-1].item()) + "\n")
  embed_file.close()  
  #test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
  #print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
      #test_loss.item(), test_micro_f1, test_macro_f1))
  torch.save(model.state_dict(),  "./model_para.pt")

if __name__ == '__main__':
  import argparse

  from utils import setup

  parser = argparse.ArgumentParser('HAN')
  parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
  parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
  parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
  args = parser.parse_args().__dict__

  args = setup(args)

  main(args)