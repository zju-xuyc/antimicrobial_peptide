import numpy as np
import copy

# 亲水性为1 疏水性为0

hydrophily = ["S","T","C","Y","N","Q","D","E","K","R","H"] # 亲水性
hydrophobe = ["G","A","V","L","I","M","F","W","P"] # 疏水性
e_charge = {"S":0,"T":0,"C":0,"Y":0,"N":0,"Q":0,"D":-1,"E":-1,"K":1,"R":1\
    ,"H":1,"G":0,"A":0,"V":0,"L":0,"I":0,"M":0,"F":0,"W":0,"P":0}

rules_list = ["0000011","0000111",
             "0001111","0011111",
             "1100000","1110000",
             "1111000","1111100",
             "1100100","0010011",
             "1001001","1001000",
             "1000100","0100100",
             "0010010","0010001",
             "0001001"]

result_list = [[] for x in rules_list]
count = 0

def get_sequence(sequence_now,charge_now,rule,rule_count):
    if rule_count==7:
        if charge_now>0:
            result_list[rules_list.index(rule)].append(sequence_now)
            
    else:
        type = int(rule[rule_count])
        if type == 1:
            for pep in hydrophily:
                get_sequence(sequence_now+pep,charge_now+e_charge[pep],rule,rule_count+1)
        else:
            for pep in hydrophobe:
                get_sequence(sequence_now+pep,charge_now+e_charge[pep],rule,rule_count+1)

for rule in rules_list:
    get_sequence("",0,rule,0)
    print(1)

file_count = 0
for result in result_list:
    f = open("./7_peptide_result/7_peptide_rule_%d.txt"%(file_count),"w")
    for r in result:
        f.write(r+"\n")
    f.close()
    file_count+=1