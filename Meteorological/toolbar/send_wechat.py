import requests


def send_wechat(title, msg):
    """
    通过pushplus推送消息到微信
    :param title: 推送标题
    :param msg: 推送内容
    """
    token = '66c20c5453fb4220b203126b4ed1a388'
    title = title
    content = msg
    template = 'html'
    url = f"https://www.pushplus.plus/send?token={token}&title={title}&content={content}&template={template}"
