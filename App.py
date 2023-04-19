from method.commonUtils import stater

BPIC2017 = 'BPIC2017.csv'
BPIC2020 = 'BPIC2020.csv'
HelpDesk = 'Help Desk.csv'
HospitalBilling = 'Hospital Billing.csv'
SepsisCases = 'Sepsis Cases.csv'

if __name__ == '__main__':
    stater(BPIC2017, k=5, threshold=2)
