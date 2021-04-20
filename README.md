# NLP Zara Challenge

## Abstract

Text summarization is the process of producing concise summaries while preserving significant information. In this paper, we investigate the issue of summarizing product titles for Zara given the product description. With the increase in the e-commerce platforms, it is necessary to compress products’ titles for consumer’s convenience. Throughout this paper, we evaluated 5 different models, and the best one turned out to be the  T5ForConditionalGeneration, a state of the art pre-trained generative model. It is an encoder-decoder model and converts all natural language processing problems into text-to-text format. However, the data at hand had a few factors that impacted the model performance, resulting in poor predictions. The product titles included some words that were not present in the description, and there were also some spelling errors in both the product name and description. We were able to tweak the model to gain predictions that were satisfactory after many iterations. 

## 1. Introduction

This paper describes the problem that the authors tried to solve when facing the Spain AI Zara Challenge. The task in this hackathon was to generate product names given the product description. This task is interesting because product name is one of the most important factors the consumer considers before making a purchase, so we wanted to make sure that using natural language processing and data-oriented solutions we were able to generate names which will adequately encompass the product’s description. However, after inspecting the data at hand, we became aware of a few factors that were going to impact our performance. First of all, in the training set of the data, we spotted some words present in the product’s name that did not appear in the product’s description. This accounted for 16\% of the tokens, and 13\% of the training observations were affected. Not only this, but the team also spotted some typos in the words conforming the product’s name and description (“with” appeared written as “whit”). Therefore, taking into account that the competition guidelines required an exact match (which included spaces, hyphens, and parentheses) to consider a prediction as a right one, we soon acknowledged that our performance in this hackathon would reach the expected standards. 

## 2. Results

Previous to the results, the team thinks that it is important for the reader to notice that for the cases where tokens occurred once or twice in the whole product description collection (e.g. _Disney_), we decided to not include then in the vocabulary, as this was biasing the results.

Generally speaking, the model was able to make predictions; in many instances, it predicted the exact word which had to be predicted just like in the test dataset yet as for the overall structure of the predicted sentence, the model failed to predict the exact match. Based on the scoring, from the ROGUE metric, we can see that the overlaps of unigrams (ROGUE-1) and bigrams (ROGUE-2) of the predictions were poor compared to the actual results. 

During the modelling we have come across the problem of overtraining. Moreover, comparing the predicted product name to the actual one, we saw that our model had difficulties predicting long text.

The rouge results from the model were:

<a href="https://www.linkpicture.com/view.php?img=LPic607f12af5061a745037501"><img src="https://www.linkpicture.com/q/rouge_scores.png" type="image"></a>

And the predictions look like:

<a href="https://www.linkpicture.com/view.php?img=LPic607f124ecd0eb1799119600"><img src="https://www.linkpicture.com/q/RESULTS.png" type="image"></a>
