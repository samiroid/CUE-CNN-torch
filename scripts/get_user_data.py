import ast
import codecs
import pandas as pd
import csv 
# csv.field_size_limit(sys.maxsize)
csv.field_size_limit(100000000)

tr_data = pd.read_csv("raw_data/data.csv")

with codecs.open("DATA/small_user_tweets.txt", "w", "utf-8") as fo:
    with open("raw_data/historical_tweets.csv", "r") as fi:
        fieldnames = ['tweet_id', 'historical_tweets']
        reader = csv.DictReader(fi)
        fieldnames = reader.fieldnames
        print(fieldnames)
        i=0
        for row in reader:            
            tweet_id = int(row[fieldnames[0]])
            historical_tweets = ast.literal_eval(row[fieldnames[1]])
            user_name = tr_data[tr_data["tweet_id"] == tweet_id]["author_full_name"]
            if len(user_name) == 0: 
                # print("tweet_id not found: {}".format(tweet_id))
                continue            
            user_name = user_name.item()            
            for t in historical_tweets[:10]:          
                ct = t.replace("\t"," ").replace("\n", " ").replace("\\n"," ")
                ct = ct.strip("\"").strip("'").split()
                ct = " ".join(ct)
                if len(ct) > 0:
                    fo.write("{}\t{}\n".format(user_name, ct))
                    # break
            # i+=1
            # if i > 100: break
            # print(tweet_id)
            # print(historical_tweets)
            
    