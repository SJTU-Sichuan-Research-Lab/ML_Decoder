import json
import urllib.request
import urllib.error


def _send(data):
    url = "https://open.feishu.cn/open-apis/bot/v2/hook/55f58a72-8f02-481e-8a9e-ba1073095af5"
    headers = {'Content-Type': 'application/json'}
    json_data = json.dumps(data).encode('utf-8')  # Encoding to bytes
    request = urllib.request.Request(url, data=json_data, headers=headers, method='POST')

    try:
        response = urllib.request.urlopen(request)
        response_data = response.read()
        print("Response:", response_data)
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code)
    except urllib.error.URLError as e:
        print("Error:", e.reason)


def feishu_bot_send_text(text):
    data = {
        "msg_type": "text",
        "content": {
            "text": text
        }
    }
    _send(data)


def feishu_bot_send_rich_text(title, text, url=None):
    content = [
        {
            "tag": "text",
            "text": text
        },
        {
            "tag": "a",
            "text": "detail",
            "href": url
        }
    ]
    if url is None:
        content = [{
            "tag": "text",
            "text": text
        }]

    data = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": title,
                    "content": [content]
                }
            }
        }
    }
    _send(data)
