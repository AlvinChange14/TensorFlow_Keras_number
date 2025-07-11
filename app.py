from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

app = Flask(__name__)

import os

from ai import AI
from file import File

# Channel Access Token
file = open('channel_access_token.txt', encoding='utf8')
text = file.read().strip()
line_bot_api = LineBotApi(text)
file.close()

# Channel Secret
file = open('channel_secret.txt', encoding='utf8')
text = file.read().strip()
handler = WebhookHandler(text)
file.close()

# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


# 處理文字訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text  # msg 是使用者發過來的String

    message = TextSendMessage(text='給我一個數字照片！')
    line_bot_api.reply_message(event.reply_token, message)
    print('-----------------')

# 處理影音訊息
@handler.add(MessageEvent, message=(ImageMessage, VideoMessage, AudioMessage))
def handle_content_message(event):
    is_image = False
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
        is_image = True
    elif isinstance(event.message, VideoMessage):
        ext = 'mp4'
    elif isinstance(event.message, AudioMessage):
        ext = 'm4a'
    else:
        return

    if is_image == False:
        line_bot_api.reply_message(event.reply_token, '這不是圖片')
    else:
        message_content = line_bot_api.get_message_content(event.message.id)
        img, file_path = file.save_bytes_image(message_content.content)
        pred = ai.predict_image_with_path(file_path)

        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text=pred)
            ])

ai = AI()
file = File()

if not os.path.exists('model/'):
    os.makedirs('model/')
if not os.path.exists('media/'):
    os.makedirs('media/')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
