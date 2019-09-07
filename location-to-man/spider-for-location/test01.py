from urllib.request import urlopen
from  bs4 import BeautifulSoup
#文字爬取
html = urlopen(r"http://www.weather.com.cn/weather/101270101.shtml")
soup = BeautifulSoup(html,features="lxml")
all_ul=soup.find_all("ul",attrs={"class":"t clearfix"})
# print(all_ul.__len__())
all_li = all_ul[0].find_all('li')
print("=========================")
for i in all_li:
    h1 = i.find("h1").get_text()
    pwea = i.find("p",attrs={"class":"wea"}).get_text()
    p2=  i.find("p",attrs={"class":"tem"})
    ptemp = p2.find("span").get_text()+"~"+p2.find("i").get_text()
    win = i.find("p",attrs={"class":"win"}).find("i").get_text()
    print(win)