model.load_state_dict(torch.load('model-99.04.pt'))
model.eval()
pred = []
label = []
with torch.no_grad():
    for methyl_t,labels_t in testdata:
        methyl_t = Variable(methyl_t.reshape([methyl_t.size()[0],1,methyl_t.size()[1]]).to(torch.float32).to(device))
        labels_t = Variable(labels_t.to(device))
        outputs = model(methyl_t)
        _,predicted = torch.max(outputs.data,1)
        _,labeled = torch.max(labels_t,1)
        pred.append(predicted.to(torch.int64).cpu().detach().numpy())
        label.append(labeled.to(torch.int64).cpu().detach().numpy())

pred = np.concatenate(pred)
label = np.concatenate(label)
cf_matrix = confusion_matrix(label, pred)
fpr, tpr, threshold = metrics.roc_curve(label, pred)
roc_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['Normal','Cancer'])
ax.yaxis.set_ticklabels(['Normal','Cancer'])
plt.show()

target_names = ['Normal', 'Cancer']
print(classification_report(label, pred, target_names=target_names))
