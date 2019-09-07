from  PIL import Image,ImageDraw
import numpy as np,random
import  os
index=414;
error=0
for i in range(520,800):

    cont = random.randint(50,200)
    paszx = random.randint(0,300-cont)
    paszy = random.randint(0,300-cont)
    type = random.randint(1,20)
    img = Image.open("E://ShuJiaPeiXun//708Spider//test_pic//pic"+str(i)+".jpg")
    # draw = ImageDraw.Draw(img)
    # draw.rectangle((123,77,217,164),outline="red",width=)
    style = img.resize((300,300))
    content = Image.open("E://ShuJiaPeiXun//708Spider//yello//"+str(type)+".png")
    content=content.resize((cont,cont))
    shape = np.shape(style)
    if len(shape)!=3:
        continue
    if(style.getbands().__len__()>3):
       continue

    r, g, b, a = content.split()
    style.paste(content,(paszx,paszy),mask=a)
    savename = r"E://ShuJiaPeiXun//708Spider//test_pic2//"+str(index)+"."+str(paszx)+"."+str(paszy)+"."+str(paszx+cont)\
               +"."+str(paszy+cont)+".jpg"
    style.save(savename)
    index+=1;
# style.show()
# draw = ImageDraw.Draw(style)
# draw.rectangle((paszx,paszy,paszx+cont,paszy+cont),outline="red")
# style.show()