from code.data import read_all_data
from code.model import model_main

train_set, train_label, valid_set, valid_label, test_set, test_label = read_all_data()

for i in range(10):
    roc_auc, prc_auc = model_main(train_set[i], train_label[i], valid_set[i], valid_label[i], test_set[i], test_label[i])
    print('# fold %d -> ROC_AUC is %f, PRC_AUC is %f' % (i, roc_auc, prc_auc))
