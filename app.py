from flask import Flask, render_template, request
import numpy as np
import requests
import json
from collections import defaultdict
import pickle
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    problems = defaultdict(int)
    ans=0
    prediction=0
    handle = ''
    friendsCount = 0
    name = ''
    maxRating = 0
    maxRank = ''
    currRating = 0
    con = 0
    photo = ''

    if request.method == "POST":
        handle = request.form['handle']
        status = 'https://codeforces.com/api/user.status?handle=' + handle
        req = requests.get(status)
        data = json.loads(req.content)
        
        if (data['status'] == 'FAILED'):
            return render_template("Predict.html", name='failed')
        if(len(data['result']) == 0):
            return render_template("Predict.html", problems=0)
        
        mx = 0
        for i in data['result']:
            if(i['verdict'] == 'OK'):
                problems[str(i['problem']['contestId']) + i['problem']['index'] + i['author']['participantType']] += 1
        cont = 'https://codeforces.com/api/user.rating?handle=' + handle
        req = requests.get(cont)
        data = json.loads(req.content)
        for i in data['result']:
            con+=1
            
        info = 'https://codeforces.com/api/user.info?handles=' + handle
        req = requests.get(info)
        data = json.loads(req.content)['result'][0]
        
        maxRating = data['maxRating']
        maxRank = data['maxRank']
        friendsCount = data['friendOfCount']
        currRating = data['rating']
        photo = data['titlePhoto']
        features=[con,maxRating,friendsCount,len(problems)]
        final_features=[np.array(features)]
        prediction=model.predict(final_features)
        
        
    return render_template("Predict.html", problems=len(problems), name=handle, maxRating=maxRating, maxRank=maxRank, friendsCount=friendsCount, currRating=currRating, contests = con, photo = photo, ans = int(currRating + prediction))


if __name__ == '__main__':
    app.run(debug=True)
