##Gráfico das curvas ROC - Franklin Magalhães Ribeiro Junior

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

url = "dadosDeTreinamento2.csv"
df = pd.read_csv(url)

X = np.array(df.drop('categoria',1))

a = np.array(df.categoria)


for i in range(len(df)):
    if (a[i]=='muito_desertico'):
        a[i]=0
    if (a[i]=='desertico'):
        a[i]=1
    if (a[i]=='critico_A'):
        a[i]=2
    if (a[i]=='quente_seco'):
        a[i]=3
    if (a[i]=='lower_fail'):
        a[i]=4
    if (a[i]=='lower_marginal'):
        a[i]=5
    if (a[i]=='lower_optimal'):
        a[i]=6
    if (a[i]=='caso_otimo'):
        a[i]=7
    if (a[i]=='upper_optimal'):
        a[i]=8    
    if (a[i]=='marginal'):
        a[i]=9
    if (a[i]=='upper_marginal'):
        a[i]=10
    if (a[i]=='upper_fail'):
        a[i]=11
    if (a[i]=='frio'):
        a[i]=12
    if (a[i]=='frio e umido'):
        a[i]=13
    if (a[i]=='frio e muito umido'):
        a[i]=14
    if (a[i]=='frio e demasiado umido'):
        a[i]=15 
        

y=[]
for i in range(len(df)):
    y.append(a[i])
      
    

# Binariza a saída para fazer os cálculos depois
y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Caĺcula ROC áreas
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        

##grafico de 1
'''
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''


####grafico de todos

# todos os false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# interpolar todas as curvas ROC
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# média da AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plota todas as curvas ROC
plt.figure()

##poderia plotar a macro average também
''' 
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='deeppink', linestyle=':', linewidth=4)
'''

colors = cycle(['gold','aqua', 'darkorange', 'cornflowerblue','pink', 'lightgreen'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='navy', linestyle='-', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="upper right", bbox_to_anchor=(2, 1))
plt.show()


