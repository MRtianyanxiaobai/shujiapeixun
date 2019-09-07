from PIL import Image,ImageDraw,ImageFont,ImageFilter
import  random

#验证码生成
#随机字母
def randChar():
    return chr(random.randint(65,90));
def randChar2():
    return str(random.randint(0,9))
#随机颜色生成
#随机颜色1
def randColor1():
    return (random.randint(64,255),
            random.randint(64, 255),
            random.randint(64, 255))

#随机颜色2
def randColor2():
    return (random.randint(32,127),
            random.randint(32, 127),
            random.randint(32, 127))
for _ in range(600):
    #创建画板
    wid = 240;
    h=60;
    img = Image.new("RGB",(wid,h),(255,255,255));
    #创建字体对象
    font = ImageFont.truetype("1.ttf",50)
    #创建Draw对象
    draw = ImageDraw.Draw(img)
    for x in range(wid):
        for y in range(h):
            draw.point((x,y),fill=randColor1());
    nums="";
    for i  in range(4):
        num = randChar2();
        nums = nums+str(num)
        draw.text((10+60*i,10), num, fill=randColor2(), font=font)
    img.filter(ImageFilter.BLUR)
    img.save("pic/"+nums+".jpg");