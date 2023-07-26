from transformers import AutoTokenizer, AutoModel
import PySimpleGUI as sg

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()

def MasterMa(model,text,history):
    model = model.eval()
    response, history = model.chat(tokenizer, "假如你是马保国老师，我是英国大理石，请问：" + text, history=history)
    return response, history

def ChatBot(model):
    layout = [[(sg.Text('马老师说：武林要以和为贵，不要搞窝里斗！', size=[40, 1]))],
              [sg.Output(size=(80, 20))],
              [sg.Multiline(size=(70, 5), enter_submits=True),
               sg.Button('偷袭一下', button_color=(sg.YELLOWS[0], sg.BLUES[0])),
               [(sg.Text('英国大理石：', size=[40, 1]))],
               sg.Button('退出聊天', button_color=(sg.YELLOWS[0], sg.BLUES[0]))]]

    window = sg.Window('马老师和英国大理石', layout, default_element_size=(30, 2))
    history = []
    while True:
        event, value = window.read()
        if event == '偷袭一下':
            text = value[1]
            response, history = MasterMa(model=model,text=text,history=history)
            print("英国大理石说：" + text)
            print("马老师教导说：" + response)
        else:
            break
    window.close()
ChatBot(model=model)
