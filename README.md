# Trial-SentAnalysis
A machine learning algorithm which takes scraped data from Supreme Court trials and an inputted hypothetical trial and outputs the outcome of the trial.


Structure:

Scraper = Takes text-scanned PDFs of SC opinions and converts them into vectorized data.
Model = Takes vectorized data and outputs learning model
    NN = Neural net (not working yet)
    LR = Logistic regression, decent precision (higher than chance)

Data is formatted like so:

{year decided, [yustice names], case name, case excerpt, case decision}


Scraper Usage: 

python scraper/scraper.py <input_directory> <output_file> <write_or_append> <max_files>

Set <max_files> to 0 if you want to scrape all files in <input_directory>

Batches of 10-25 case files recommended, for older volumes you will need to double check scraped results. 
Justices may be erroneous, e.g. "stewart" may appear as "art". 

Model Usage:

python model/<model>.py

For every model not transformer.py, you must specify the <input_file> like so:

python model/<model>.py <input_file>

For the lrmodel, you must specify whether you split temporally or not. a 1 signifies temporal split, a 0 signifies random:

python model/<model>.py <input_file> <random/temporal>


Prediction:

python model/<predictor>.py

You may predict using the case_input.jsonl file. Input the case you want to predict in the proper JSONL format.