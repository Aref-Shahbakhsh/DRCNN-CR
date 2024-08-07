model.load_state_dict(torch.load('model-ft-89.81.pt'))
model.eval()
pred_t = []
label_t = []
with torch.no_grad():
    for methyl_test,labels_test in test_dataload:
        methyl_test = Variable(methyl_test.reshape([methyl_test.size()[0],1,
                                                    methyl_test.size()[1]]).to(torch.float32).to(device))
        labels_test = Variable(labels_test.to(device))
        outputs = model(methyl_test)
        _,t_predicted = torch.max(outputs.data,1)
        _,t_labeled = torch.max(labels_test,1)
        pred_t.append(t_predicted.to(torch.int64).cpu().detach().numpy())
        label_t.append(t_labeled.to(torch.int64).cpu().detach().numpy())
        
pred_t = np.concatenate(pred_t)
label_t = np.concatenate(label_t)
cf1_matrix = confusion_matrix(label_t, pred_t)
fpr_t, tpr_t, threshold_t = metrics.roc_curve(label_t, pred_t)
roc_auc_t = metrics.auc(fpr_t, tpr_t)

plt.plot(fpr_t, tpr_t, 'b', label = 'AUC = %0.2f' % roc_auc_t)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


ax = sns.heatmap(cf1_matrix, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['Normal','Cancer'])
ax.yaxis.set_ticklabels(['Normal','Cancer'])
plt.show()

target_names = ['Normal', 'Cancer']
print(classification_report(label_t, pred_t, target_names=target_names))