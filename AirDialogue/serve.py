from flask import Flask, request
from flask_cors import CORS
import argparse
import base64
import pickle as pkl
import time
from seq2seq.database import *
from seq2seq.util.checkpoint import Checkpoint
import json
import redis
import traceback
import multiprocessing as mp

Q = None

app = Flask(__name__)
CORS(app)

r = redis.Redis(host='localhost', port=6379, db=2)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/message', methods=['GET', 'POST'])
def respond():
    if request.method == 'GET':
        history = request.args.get('history', None)
        scenario_idx = int(request.args.get('scenario_idx', None))
    else:
        history = request.form.get('history', None)
        scenario_idx = int(request.form.get('scenario_idx', None))

    history = json.loads(str(base64.b64decode(history)))
    print('[DEBUG] History recieved:', history)
    # generate response
    request_id = int(r.incr('request_id_counter'))
    print('[DEBUG] queueing message with request id:', request_id)
    Q.put((request_id, scenario_idx, history,))
    while not r.exists("result_%d" % (request_id)):
        time.sleep(0.05)
    print('[DEBUG] de-queueing message with request id:', request_id)
    result = pkl.loads(r.get("result_%d" % (request_id)))
    r.delete("result_%d" % (request_id))
    print('[DEBUG] Response:', result)
    return json.dumps(result)

def _chatbot_f(converse_obj, args, model, data_list, scenario_idx, history):
    return converse_obj.reply_one(args, model, data_list, scenario_idx, history)

def model_process(converse_obj, args, model, data_list):
    print('CHATBOT LOADED!')
    while True:
        try:
            request_id, scenario_idx, history = Q.get()
            result = _chatbot_f(converse_obj, args, model, data_list, scenario_idx, history)
            r.set('result_%d' % (request_id), pkl.dumps(result))
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

def flask_process(port):
    app.run(host='0.0.0.0', port=port, threaded=True, processes=1)

def main(converse_obj, args, model, dataloader):
    global Q

    flask_args = argparse.Namespace()
    flask_args.port = 5000

    data_list = list(dataloader)
    latest_checkpoint_path = Checkpoint.get_latest_checkpoint(converse_obj.model_dir)
    resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
    model.load_state_dict(resume_checkpoint.model)
    converse_obj.optimizer = None
    converse_obj.args = args
    model.args = args

    q = mp.Manager().Queue()
    Q = q

    p = mp.Process(target=flask_process, args=(flask_args.port,))
    p.start()

    model_process(converse_obj, args, model, data_list)


if __name__ == "__main__":
    main()
