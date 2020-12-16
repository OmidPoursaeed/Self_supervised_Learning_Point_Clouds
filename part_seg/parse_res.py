acc = []
iou = []
tot = []
with open('test_res.txt') as fr:
    for eachLine in fr:
        items = eachLine.lstrip(' ').rstrip('\n').split(' ')
        if (items[1] == 'Accuracy:'):
            acc.append(float(items[2]))
        if items[1] == 'IoU:':
            iou.append(float(items[2]))
        if (items[1] == 'Total'):
            tot.append(int(items[3]))

for i in acc:
    print(i)
print('\n')
for i in iou:
    print(i)
print('\n')
for i in tot:
    print(i)