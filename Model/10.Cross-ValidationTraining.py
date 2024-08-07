list_epoch = []
list_accuracy = []
list_loss = []
list_loss_train = []
num_epochs = 300
itr = 0
for epoch in range(num_epochs):
    for i,(methyl,labels) in enumerate(traindata):
        methyl = Variable(methyl.reshape([methyl.size()[0],1,methyl.size()[1]]).to(torch.float32).to(device))
        labels = Variable(labels.to(torch.float32).to(device))

        optimizer.zero_grad()

        outputs = model(methyl)

        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        itr += 1
        if itr % (6) == 0:
            best_accuracy = 0
            correct = 0
            total = 0
            for methyl_t,labels_t in testdata:
                methyl_t = Variable(methyl_t.reshape([methyl_t.size()[0],1,methyl_t.size()[1]]).to(torch.float32).to(device))
                labels_t = Variable(labels_t.to(device))
                outputs = model(methyl_t)
                _,predicted = torch.max(outputs.data,1)
                _,labeled = torch.max(labels_t,1)
                total += labels_t.size(0)
                correct += (predicted.cpu() == labeled.cpu()).sum()
            accuracy = 100 * correct / total

            list_epoch.append(itr)
            list_accuracy.append(accuracy)
            if accuracy >= max(list_accuracy):
                round_accuracy = round(accuracy.tolist(),2)
                torch.save(model.state_dict(), f'model-{round_accuracy}.pt')
            list_loss.append(loss.cpu().data.numpy())
            print(f"Iteration: {itr}. loss: {loss.data}. Accuracy: {accuracy}")

    scheduler(epoch)


plt.figure(figsize=[15, 8])
plt.plot(list_accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Validation Accuracy')
plt.show()

plt.figure(figsize=[15, 8])
plt.plot(list_loss, color = 'orange')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Validation Loss') 
plt.show()

print('Validation Accuracy:',np.mean(list_accuracy), 'Validation Loss:', np.mean(list_loss))