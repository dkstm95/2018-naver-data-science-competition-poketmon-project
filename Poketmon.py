import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import csv
from pandas import Series, DataFrame

new_total = []
num_High = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
num_Mid = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
num_Low = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
num_lgd_High = 0
num_lgd_Mid = 0
num_lgd_Low = 0
num_lgd = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
strong_pok = []
weak_pok = []
sw_pair = {'HP': 0, 'Attack': 0, 'Defense': 0, 'Sp. Atk': 0, 'Sp. Def': 0, 'Speed': 0}
ws_pair = {'HP': 0, 'Attack': 0, 'Defense': 0, 'Sp. Atk': 0, 'Sp. Def': 0, 'Speed': 0}
sw_100_pair = {'HP': 0, 'Attack': 0, 'Defense': 0, 'Sp. Atk': 0, 'Sp. Def': 0, 'Speed': 0}
ws_100_pair = {'HP': 0, 'Attack': 0, 'Defense': 0, 'Sp. Atk': 0, 'Sp. Def': 0, 'Speed': 0}
feature = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

data = pd.read_csv('Pokemon.csv')

data_rating = data.pivot_table('Total', index='HP', columns='Generation', aggfunc='mean')
data_rating.head()

for i in range(800):
    new_sum = (int(data['HP'][i]) / 255 * 17) + (int(data["Attack"][i]) / 190 * 17) \
              + (int(data["Defense"][i]) / 230 * 17) + (int(data["Sp. Atk"][i]) / 194 * 16) \
              + (int(data["Sp. Def"][i]) / 230 * 16) + (int(data["Speed"][i]) / 180 * 17)
    new_total.append(int(new_sum))

    if new_sum >= 49:
        num_High[data['Generation'][i]] = num_High[data['Generation'][i]] + 1
        if data['Legendary'][i] == True:
            num_lgd_High = num_lgd_High + 1
        strong_pok.append(i)
    elif new_sum <= 19:
        num_Low[data['Generation'][i]] = num_Low[data['Generation'][i]] + 1
        if data['Legendary'][i] == True:
            num_lgd_Low = num_lgd_Low + 1
        weak_pok.append(i)
    else:
        num_Mid[data['Generation'][i]] = num_Mid[data['Generation'][i]] + 1
        if data['Legendary'][i] == True:
            num_lgd_Mid = num_lgd_Mid + 1
    if data['Legendary'][i] == True:
        num_lgd[data['Generation'][i]] = num_lgd[data['Generation'][i]] + 1

data['New_Total'] = new_total

sum1 = 0
sum2 = 0
sum3 = 0
for i in range(1,7):
    sum1 = sum1 + num_High[i]
    sum2 = sum2 + num_Mid[i]
    sum3 = sum3 + num_Low[i]

print(sum1, sum2, sum3)
#2번 그래프 (total별 포켓몬 수)
sns.countplot(x=new_total, data=data)
plt.title('distribution of the poketmon by total score')
plt.xlabel('total score')
plt.ylabel('number of poketmon')
plt.show()

#3번 그래프 (세대별 상 중 하 포켓몬 비율)
num_High = pd.Series(num_High)
num_Mid = pd.Series(num_Mid)
num_Low = pd.Series(num_Low)
#상 그래프
plt.pie(num_High, labels=num_High.index, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage of High rank by Generation')
plt.show()
#중 그래프
plt.pie(num_Mid, labels=num_Mid.index, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage of Middle rank by Generation')
plt.show()
#하 그래프
plt.pie(num_Low, labels=num_Low.index, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage of Low rank by Generation')
plt.show()

#4번 그래프 (전설 포켓몬 상 중 하 비율)
num_lgd = pd.Series(num_lgd)
xs = ["High", "Middle", "Low"]
ys = [num_lgd_High, num_lgd_Mid, num_lgd_Low]
#레전드 포켓몬의 세대별 수
plt.pie(num_lgd, labels=num_lgd.index, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage of Legendary Poketmon by Generation')
plt.show()
#레전드 포켓몬의 상중하 랭크의 비율
plt.bar(xs, ys)
plt.xlabel("Rank")
plt.ylabel("Number of Legend Poketmon")
plt.show()

for index1 in strong_pok:
    for index2 in weak_pok:
        for j in feature:
            if data[j][index1] > data[j][index2]:
                sw_pair[j] = sw_pair[j] + 1
            else:
                ws_pair[j] = ws_pair[j] + 1

for i in feature:
    sw_100_pair[i] = int(sw_pair[i] / (sw_pair[i]+ws_pair[i]) * 100)
    ws_100_pair[i] = int(ws_pair[i] / (sw_pair[i]+ws_pair[i]) * 100)

#각 feature별 strong, weak 표
pyo = DataFrame([sw_pair, ws_pair, sw_100_pair, ws_100_pair], index=['strong>weak', 'weak>strong', 'strong>weak(%)', 'weak>strong(%)'])
print(pyo)

f = open('output.csv', 'w', newline='')
wr = csv.writer(f)
wr.writerow(['#', 'Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary', 'New_Total', 'flag'])
for i in range(800):
    if data['New_Total'][i] >= 49:
        wr.writerow([i,data['Name'][i],data['Type 1'][i],data['Type 2'][i],data['Total'][i],data['HP'][i]\
                     ,data['Attack'][i],data['Defense'][i],data['Sp. Atk'][i],data['Sp. Def'][i],data['Speed'][i]\
                     ,data['Generation'][i],data['Legendary'][i],data['New_Total'][i],1.0])
    elif data['New_Total'][i] <= 19:
        wr.writerow([i,data['Name'][i],data['Type 1'][i],data['Type 2'][i],data['Total'][i],data['HP'][i]\
                        ,data['Attack'][i],data['Defense'][i],data['Sp. Atk'][i],data['Sp. Def'][i],data['Speed'][i] \
                        , data['Generation'][i], data['Legendary'][i], data['New_Total'][i], 0.0])
#강한 포켓몬은 flag를 1로 해놓았고, 약한 포켓몬은 flag를 0으로 해놓았다.
f.close()

data2 = pd.read_csv('output.csv')
train_cols = data2.columns[5:11]
logit = sm.Logit(data2['flag'], data2[train_cols])
result = logit.fit()
print(result.summary())
'''여기서 주목할 건 coef(편회귀계수)라는 거래. 이 값이 양수이면 그 column의 값이 커질수록 flag가 1일 확률이
높아지고 반대로 값이 음수이면 그 column의 값이 커질수록 flag가 0일 확률이 높아진대.
그럼 여기서 HP의 coef가 -0.0495이므로 음수야. 그럼 HP가 높을수록 약한 포켓몬일 가능성(flag==0)이 높대....(왜지?)
암튼 그리고 Attack의 coef가 0.0415므로 양수야. 그럼 Attack이 높을수록 강한 포켓몬(flag==1)일 가능성이 높대!'''

data2['predict'] = result.predict(data[train_cols])
print(data2.head())
'''여기서 predict가 뭐냐면 회귀모델을 통해 각 feature의 값으로 flag의 값을 예측하는거야.
그래서 예를들어 실제로는 피카츄가 flag가 1인데(강한 포켓몬) 예측값(predict)은 feature(HP,Attack 요런거)를 
지가 알아서 weight를 임의로 해서 계산했더니 0.9816이 나왔다. 뭐 이런거야. 
나는 feature를 줄테니 컴퓨터야 요 포켓몬이 강한놈인지 약한놈인지 계산해보거라 이런거....암튼 정신이 없다 지금'''


'''솔직히 지금 새벽 4시인데 예시자료에 나와있는 것처럼 strong이 weak보다 큰 pair 수 뭐 이렇게 나눠서 
회귀모델 돌리는거 지금 내 정신과 머리로는 불가능한 것 같애. 그래서 최대한 할 수 있는 만큼 한게 각 feature가 flag에 
어떤 영향을 미치는가 계산한거야. 나의 한계는 여기까지 인거 같구 보고서랑 나머지 첨삭만 부탁할게...'''

'''참고 : http://3months.tistory.com/28?category=753896'''

fire = data[(data['Type 1'] == 'Fire') | ((data['Type 2']) == "Fire")]
water = data[(data['Type 1'] == 'Water') | ((data['Type 2']) == "Water")]
plt.scatter(fire.Attack.head(50), fire.Defense.head(50), color='R', label='Fire', marker="*", s=50)
plt.scatter(water.Attack.head(50), water.Defense.head(50), color='B', label="Water", s=25)
plt.xlabel("Attack")
plt.ylabel("Defense")
plt.legend()
plt.plot()
fig = plt.gcf()  #get the current figure using .gcf()
fig.set_size_inches(12, 6) #set the size for the figure
plt.show()

#type1별로 가장 강한 포켓몬
strong = data.sort_values(by='New_Total', ascending=False)
strong.drop_duplicates(subset=['Type 1'], keep='first')
print(strong)

#type별 포켓몬 분포도
labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'
sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]
colors = ['Y', 'B', '#00ff00', 'C', 'R', 'G', 'silver', 'white', 'M']
explode = (0.1, 0, 0.1, 0, 0., 0, 0, 0, 0)  # only "explode" the 3rd slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title("Percentage of Different Types of Pokemon")
plt.plot()
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.show()

#type별 Attack 분포도, 원하는 속성으로 교체 가능(y 변수 대입값만 바꿔주면 됨, x도 마찬가지)
plt.subplots(figsize=(15, 5))
plt.title('Attack by Type 1')
sns.boxplot(x="Type 1", y="HP", data=data)
plt.ylim(0, 200)
plt.show()

#legend 포켓몬이 다른 포켓몬에 비해 얼마나 강한 스탯을 갖고 있는지(New_Total 기준, 이것도 다른 속성 변수로 대체 가능함) 한눈에 파악 가능
plt.figure(figsize=(12, 6))
top_types = data['Type 1'].value_counts()[:10] #take the top 10 Types
df1 = data[data['Type 1'].isin(top_types.index)] #take the pokemons of the type with highest numbers, top 10
sns.swarmplot(x='Type 1', y='New_Total', data=df1, hue='Legendary') # this plot shows the points belonging to individual pokemons
# It is distributed by Type
plt.axhline(df1['New_Total'].mean(), color='red', linestyle='dashed')
plt.show()

#heatmap, 말 그대로 더 높은 수치일수록 뜨거워지는 표. 속성들간의 상관관계를 알기 쉬움
plt.figure(figsize=(10,6)) #manage the size of the plot
sns.heatmap(data.corr(), annot=True) #df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap
plt.show()

#세대별로, 각 type 포켓몬의 수를 꺾은선 그래프로 표현한 것.
a = data.groupby(['Generation','Type 1']).count().reset_index()
a = a[['Generation', 'Type 1', 'New_Total']]
a = a.pivot('Generation', 'Type 1', 'New_Total')
a[['Water', 'Fire', 'Grass', 'Dragon', 'Normal', 'Rock', 'Flying', 'Electric']].plot(color=['b', 'r', 'g', '#FFA500', 'brown', '#6666ff', '#001012', 'y'], marker='o')
fig=plt.gcf()
fig.set_size_inches(12, 6)
plt.show()
