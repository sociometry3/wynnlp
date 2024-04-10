from liteModel import doPredict
from string import Template
import os
from sanic import Sanic
from sanic.response import json
from cors import add_cors_headers
from options import setup_options
# from sanic_ext import Extend

app = Sanic("AVAPredict")
app.config.CORS_ORIGINS = "http://10.32.203.241:51980"
# Extend(app)
async def predict(request):
    print(request.json)
    if (request.json) :
        text = request.json.get("text")
        result = doPredict(text)
        return json(result)
    return json({})
async def record(request):
    if (request.json):
        item = request.json.get("text")
        with open('./data/record.json', 'r', encoding="utf8") as rf:
            data = rf.read()
        with open('./data/record.json', 'w', encoding="utf8") as rf:
            data += (item + '\n')
            rf.write(data)
        # text = request.json.get("text")
        # spo_list = request.json.get("spo_list")
        # with open('./data/record.json', 'w') as rf:
        #     data = rf.read()
        #     s = Template("{\"$text\":\"$spo_list\"}")
        #     d = {
        #         text,
        #         spo_list,
        #     }
        #     data += s.substitute(d)
        #     rf.write(data)
        return json({})
    return json({})
app.add_route(predict, '/predict', ['POST', 'OPTIONS'])
app.add_route(record, '/record', ['POST', 'OPTIONS'])
# @app.post("/predict")
# async def predict(request):
#     print(request.json)
#     text = request.json.get("text")
#     result = doPredict(text)
#     return json(result)

app.register_listener(setup_options, "before_server_start")
app.register_middleware(add_cors_headers, "response")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=52002, fast=False)
