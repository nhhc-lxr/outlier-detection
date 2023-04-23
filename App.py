from method.CommonUtils import stater

# 这两个数据集质量比较高，效果差距不大，lof算法F1值最高，效率稍低
BPIC2017 = 'BPIC2017.csv'
BPIC2020 = 'BPIC2020.csv'
# 这个也还行
HelpDesk = 'Help Desk.csv'
# 这个数据集各方面效果都很差，完全比不过dbscan，最好不要写进去，或者只用谱聚类作为参考，放弃dbscan对比，再或者你可以尝试自己调参
HospitalBilling = 'Hospital Billing.csv'
# 这个数据集你可以看一下统计数据，存在大量频率为1的轨迹，dbscan的效果受bleu_eps影响， 你可以把这个值调整一下，会让dbscan效果好很多
# 反正lof算法效果还不错，只是效率比较低
SepsisCases = 'Sepsis Cases.csv'

if __name__ == '__main__':
    # 上面5个变量表示data/realLog下的五个数据集
    stater(BPIC2017)
    stater(BPIC2020)
    stater(HelpDesk)
    # stater(HospitalBilling)
    # stater(SepsisCases)
