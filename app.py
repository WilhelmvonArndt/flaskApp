#!/usr/bin/env python3

from flask import Flask, render_template, request, redirect, url_for, make_response, abort
from markupsafe import escape
import pymongo
import datetime
from bson.objectid import ObjectId
import os
import subprocess
from random import sample

# instantiate the app
app = Flask(__name__)

# load credentials and configuration options from .env file
# if you do not yet have a file named .env, make one based on the template in env.example
import credentials
config = credentials.get()

# turn on debugging if in development mode
if config['FLASK_ENV'] == 'development':
    # turn on debugging, if in development
    app.debug = True # debug mnode

# make one persistent connection to the database
connection = pymongo.MongoClient(config['MONGO_HOST'], 27017, 
                                username=config['MONGO_USER'],
                                password=config['MONGO_PASSWORD'],
                                authSource=config['MONGO_DBNAME'])
db = connection[config['MONGO_DBNAME']] # store a reference to the database

# set up the routes

@app.route('/')
def home():
    """
    Route for the home page
    """
    return render_template('index.html')


# @app.route('/read') #rewrite to sort on rankings
# def read():
#     """
#     Route for GET requests to the read page.
#     Displays some information for the user with links to other pages.
#     """
#     docs = db.exampleapp.find({}).sort("created_at", -1) # sort in descending order of created_at timestamp
#     return render_template('read.html', docs=docs) # render the read template


@app.route('/create')
def create():
    """
    Route for GET requests to the create page.
    Displays a form users can fill out to create a new document.
    """
    return render_template('create.html') # render the create template

@app.route('/about')
def about():
    return render_template('about.html') 

@app.route('/create', methods=['POST'])
def create_post(): #NOTE this is edited
    """
    Route for POST requests to the create page.
    Accepts the form submission data for a new document and saves the document to the database.
    """
    name = request.form['fname']
    link = request.form['flink'] #where the user will enter the link to their animalpic

    if name == "Nimda" and link == "Nimda":
            return redirect(url_for('admin')) 

    else:

        # create a new document with the data the user entered
        doc = {
            "name": name, #Offers user oppurtunity to enter name of animal
            "links": link,
            "count": 0,
            "created_at": datetime.datetime.utcnow()
        }
        db.exampleapp.insert_one(doc) # insert a new document, This works

        return redirect(url_for('home')) 

@app.route('/edit/<mongoid>')
def edit(mongoid):
    """
    Route for GET requests to the edit page.
    Displays a form users can fill out to edit an existing record.
    """
    doc = db.exampleapp.find_one({"_id": ObjectId(mongoid)})
    return render_template('edit.html', mongoid=mongoid, doc=doc) # render the edit template


# @app.route('/edit/<mongoid>', methods=['POST'])
# def edit_post(mongoid):
#     """
#     Route for POST requests to the edit page.
#     Accepts the form submission data for the specified document and updates the document in the database.
#     """
#     name = request.form['fname']
#     message = request.form['fmessage']

#     doc = {
#         # "_id": ObjectId(mongoid), 
#         "name": name, 
#         "message": message, 
#         "created_at": datetime.datetime.utcnow()
#     }

#     db.exampleapp.update_one(
#         {"_id": ObjectId(mongoid)}, # match criteria
#         { "$set": doc }
#     )

#     return redirect(url_for('read')) # tell the browser to make a request for the /read route


@app.route('/delete/<mongoid>')
def delete(mongoid):
    """
    Route for GET requests to the delete page.
    Deletes the specified record from the database, and then redirects the browser to the read page.
    """
    db.exampleapp.delete_one({"_id": ObjectId(mongoid)})
    return redirect(url_for('read')) # tell the web browser to make a request for the /read route.

@app.route('/webhook')
def webhook():
    """
    GitHub can be configured such that each time a push is made to a repository, GitHub will make a request to a particular web URL... this is called a webhook.
    This function is set up such that if the /webhook route is requested, Python will execute a git pull command from the command line to update this app's codebase.
    You will need to configure your own repository to have a webhook that requests this route in GitHub's settings.
    Note that this webhook does do any verification that the request is coming from GitHub... this should be added in a production environment.
    """
    # run a git pull command
    process = subprocess.Popen(["git", "pull"], stdout=subprocess.PIPE)
    pull_output = process.communicate()[0]
    # pull_output = str(pull_output).strip() # remove whitespace
    process = subprocess.Popen(["chmod", "a+x", "flask.cgi"], stdout=subprocess.PIPE)
    chmod_output = process.communicate()[0]
    # send a success response
    response = make_response('output: {}'.format(pull_output), 200)
    response.mimetype = "text/plain"
    return response

@app.errorhandler(Exception)
def handle_error(e):
    """
    Output any errors - good for debugging.
    """
    return render_template('error.html', error=e) # render the edit template

#Added by me
@app.route('/rankings')
def rankings():
    """
    Route for rankings page
    """
    return render_template('rankings.html')


def get_random_docs():
    docs = list(db.exampleapp.find({}))
    
    # randomly select two documents
    random_docs = sample(docs, 2)
    return random_docs

#BELOW IS A TRIAL FUNCTION
def get_random_docs1():

    docs = list(db.exampleapp.find({}))
    random_docs = sample(docs, 2)

    doc1, doc2 = docs[:2]
    return doc1, doc2

@app.route('/vote_animal/<mongoid>')
def vote_animal(mongoid):
    db.exampleapp.update_one({"_id": ObjectId(mongoid)}, {"$inc": {"count": 1}})
    return redirect(url_for('home'))

@app.route('/delete_animal/<mongoid>')
def delete_animal(mongoid):
    db.exampleapp.delete_one({"_id": ObjectId(mongoid)})
    return redirect(url_for('admin'))


app.jinja_env.globals.update(get_random_docs=get_random_docs)
app.jinja_env.globals.update(get_random_docs1=get_random_docs1)


def get_rankings():

    top_docs = db.exampleapp.find().sort('count', -1).limit(5)
    rankings = []
    for doc in top_docs:
        rankings.append({
            'link': doc['links'],
            'count': doc['count']
        })

    return rankings

@app.route('/admin')
def admin():
    docs = db.exampleapp.find().sort('created_at', -1)
    animals = []
    for doc in docs:
        animals.append({
            'link': doc['links'],
            'name': doc['name']
        })

    referrer = request.referrer # get the URL of the referring page
    print(referrer)
    if referrer != 'https://i6.cims.nyu.edu/~wv2016/databases/web-app-DataWizardLXIX/flask.cgi/create': # replace with the URL of the page that should lead to restricted page
        return render_template('index.html')
    else:
        return render_template('admin.html')

app.jinja_env.globals.update(get_rankings=get_rankings)

def get_all():
    docs = list(db.exampleapp.find().sort('created_at', -1))
    return docs

app.jinja_env.globals.update(get_all=get_all)

if __name__ == "__main__":
    #import logging
    #logging.basicConfig(filename='/home/ak8257/error.log',level=logging.DEBUG)
    app.run(debug = True)
