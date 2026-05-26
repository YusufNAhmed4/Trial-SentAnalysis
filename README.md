# Trial-SentAnalysis
A machine learning algorithm which takes scraped data from Supreme Court trials and an inputted hypothetical trial and outputs the outcome of the trial.


Structure:

Scraper = Takes text-scanned PDFs of SC opinions and converts them into vectorized data.
Model = Takes vectorized data and outputs learning model
    NN = Neural net (not working yet)
    LR = Logistic regression, decent precision (higher than chance)

Data is formatted like so:

{year decided, [yustice names], case name, case excerpt, case decision}

Batches of 10-50 case files recommended, for older volumes you will need to double check scraped results. Justices may not show up and case excerpts may be erroneous. 