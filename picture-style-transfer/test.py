from torchvision import transforms
from run_code import run_style_transfer
from load_img import load_img, show_img
from torch.autograd import Variable

style_img = load_img('./picture/style2.jpg')
style_img = Variable(style_img).cpu()
content_img = load_img('./picture/content2.jpg')
content_img = Variable(content_img).cpu()

input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=200)

show_img(out.cpu())
save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
save_pic.save('./picture/saved_picture2.jpg')