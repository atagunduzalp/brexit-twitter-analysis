# brexit-tweet analysis

# Summary #
This project is about *text engineering* and *machine learning operations*. The topic of this work is, to present the relation between Twitter and Brexit. For training, tweets are taken from some politicians' tweets which include *Brexit* words between 2015-04-30 and 2017-04-30. After the model created, to test the model, randomly 10.000 tweets taken from Twitter API between 2015-04-30 - 2016-04-30. You can take a look brief presentation and results in *Twitter - Brexit Analysis.pdf* file. 
### Steps ###

1. First, tweets taken from Twitter API and collected in a CSV file. -> You can take a look at getTweets.py
2. Then these tweets classified by hand as positive negative and notr. *original-tweets.csv* is classified tweets that is going to train. In classes; 1 represents for *want to exit*, 0 represents for *want to stay* and 2 represents for *inactive*. 
3. Data cleaning operations takes places in *data_cleaner.py* . data_cleaner.py file is also main class for the project.
4. Last steps for training operations in *model.py* file. 
5. And then, finally, model can be tested with *test.py* file.
