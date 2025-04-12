# import numpy as np
# from metrics import evaluate
#
#
#
# Label0 = np.load('D:/result/site0_fold0_Label.npy')
# Label1 = np.load('D:/result/site0_fold1_Label.npy')
# Label2 = np.load('D:/result/site0_fold2_Label.npy')
# Label3 = np.load('D:/result/site0_fold3_Label.npy')
# Label4 = np.load('D:/result/site0_fold4_Label.npy')
#
# Pred0 = np.load('D:/result/site0_fold0_Pred.npy')
# Pred1 = np.load('D:/result/site0_fold1_Pred.npy')
# Pred2 = np.load('D:/result/site0_fold2_Pred.npy')
# Pred3 = np.load('D:/result/site0_fold3_Pred.npy')
# Pred4 = np.load('D:/result/site0_fold4_Pred.npy')
#
#
# Prob0 = np.load('D:/result/site0_fold0_Prob.npy')
# Prob1 = np.load('D:/result/site0_fold1_Prob.npy')
# Prob2 = np.load('D:/result/site0_fold2_Prob.npy')
# Prob3 = np.load('D:/result/site0_fold3_Prob.npy')
# Prob4 = np.load('D:/result/site0_fold4_Prob.npy')
#
# # print(Prob4.shape)
#
# for i in range(4):
#     evaluation_metrics = evaluate(pred=Pred0, prob=Prob0, label=Label0, acc_only=False)
#
# print(evaluation_metrics)



import numpy as np
from metrics import evaluate
# f'D:/result/site0_fold{i}_Label.npy'
# 加载数据
labels = [
    # np.load(f'D:/2024博士工作/FL/code/FLcode/result mdd/site2_fold{i}_Label.npy') for i in range(5)
    np.load(f'D:/result-abide/site1_fold{i}_Label.npy') for i in range(5)
]
preds = [
    np.load(f'D:/result-abide/site1_fold{i}_Pred.npy') for i in range(5)
]
probs = [
    np.load(f'D:/result-abide/site1_fold{i}_Prob.npy') for i in range(5)
]

# 用于存储每次的评估结果
evaluation_results = []

# 评估每个折的数据
for i in range(5):
    evaluation_metrics = evaluate(pred=preds[i], prob=probs[i], label=labels[i], acc_only=False)
    evaluation_results.append(evaluation_metrics)

# 转换成 NumPy 数组便于计算
evaluation_results = np.array(evaluation_results)


# 计算均值与方差
mean_metrics = np.mean(evaluation_results, axis=0)
std_metrics = np.std(evaluation_results, axis=0)

# 打印结果
print("Evaluation Metrics for Each Fold:")
for i, metrics in enumerate(evaluation_results):
    print(f"Fold {i}: {metrics}")

print("\nMean of Metrics Across Folds:")
print(mean_metrics)

print("\nStandard Deviation of Metrics Across Folds:")
print(std_metrics)

