import numpy as np

ckpt_no = 25
TAU = 10
epoch = 100
gnn_type = 'gin'
best_val_results = {}
results = {}
avg_result = []
for ds in ['bbbp', 'clintox', 'muv', 'hiv', 'bace', 'sider', 'tox21', 'toxcast']:
    try:
        with open(f'./results-{gnn_type}-{TAU}-{ckpt_no}-{epoch}/{ds}/result.txt', 'r') as f:
            lines = f.readlines()
        best_val_test_acc = []
        test_acc = []
        for line in lines:
            best_val_test_acc.append(100 * float(line.strip('\n').split()[-3]))
            test_acc.append(100 * float(line.strip('\n').split()[-1]))
        best_val_results.update(
            {f'{ds}': f'{np.mean(best_val_test_acc):.2f} / {np.std(best_val_test_acc):.2f}'}
        )
        results.update(
            {f'{ds}': f'{np.mean(test_acc):.2f} / {np.std(test_acc):.2f}'}
        )
        avg_result.extend(test_acc)
    except:
        best_val_results.update({f'{ds}': np.nan})
        results.update({f'{ds}': np.nan})
        avg_result.extend([np.nan])

print('#########################')
for k, v in results.items():
    print(f'{k:<8}: {v:<15}')
print('#########################')
avg_result = np.array(avg_result)
print(f'Average Result: {np.mean(avg_result[~np.isnan(avg_result)]):.2f}')
print('#########################')
# for k, v in best_val_results.items():
#     print(f'{k:<8}: {v:<15}')
