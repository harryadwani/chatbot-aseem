# chatbot-aseem.   
Python chatbot to answer generic questions and specific questions for the platform aseem.abhyast.in .

How to run:
npm install -g localtunnel
lt --port 5000 --subdomain <any-subdomain>

To train:
python train.py (after editing intents.json)

Use batch=10, epoch=200 latest for tried and tested results.

Update links in intents if training for your own domain.
